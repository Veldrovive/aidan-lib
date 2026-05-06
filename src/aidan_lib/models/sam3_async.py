import uuid
import contextlib
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory, resource_tracker
import numpy as np
from PIL import Image
import imageio.v3 as iio
from pathlib import Path
from multiprocessing.connection import Listener, Client
import asyncio
from .sam3_base import DEFAULT_SAM3_SERVER_ADDRESS, SAM3VideoOutput, SAM3Harness, _get_buffer_specs


class SAM3HarnessServer:
    def __init__(
        self,
        max_num_frames: int,
        max_frame_width: int,
        max_frame_height: int,
        frame_dtype: np.dtype,
        segmentation_dtype: np.dtype, # Assume SEG_ID_TYPE is passed here
        address: tuple[str, int] | None = None,
        segmenter=None,
        segmenter_kwargs: dict | None = None,
        max_concurrent_sessions: int = 4 # Adjust based on CPU cores / VRAM
    ):
        self.address = address if address is not None else DEFAULT_SAM3_SERVER_ADDRESS
        self.max_num_frames = max_num_frames
        self.max_frame_width = max_frame_width
        self.max_frame_height = max_frame_height
        self.frame_dtype = frame_dtype
        self.segmentation_dtype = segmentation_dtype

        # Get buffer shapes to reconstruct arrays dynamically later
        _, self.frame_buffer_shape = _get_buffer_specs(
            self.max_num_frames, self.max_frame_width, self.max_frame_height, 3, self.frame_dtype
        )
        _, self.segmentation_buffer_shape = _get_buffer_specs(
            self.max_num_frames, self.max_frame_width, self.max_frame_height, 1, self.segmentation_dtype
        )

        if segmenter is None:
            segmenter_kwargs = segmenter_kwargs or {}
            print(f"[SAM3 SERVER] Initializing SAM3Harness with kwargs: {segmenter_kwargs}")
            # Initialize your actual model here
            segmenter = SAM3Harness(**segmenter_kwargs)
            print(f"[SAM3 SERVER] Created multiplex predictor")

        self.segmenter = segmenter
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_sessions)

    def _start_session(self, num_frames: int, height: int, width: int, frame_numbers: list[int], shm_names: dict, offload_state_to_cpu: bool | None = None, store_session: bool = True):
        assert num_frames <= self.max_num_frames, f"Frames {num_frames} exceeds max {self.max_num_frames}."
        
        # 1. Temporarily attach to the client's shared memory
        frame_shm = shared_memory.SharedMemory(name=shm_names["frame"])
        try:
            frame_array = np.ndarray(self.frame_buffer_shape, dtype=self.frame_dtype, buffer=frame_shm.buf)
            frame_chunk = frame_array[:num_frames, :height, :width]
            
            # 2. CPU Heavy: Convert to PIL
            frame_chunk_pil = [Image.fromarray(frame) for frame in frame_chunk]
        finally:
            frame_shm.close() # Always close the reference when done reading

        # 3. Pass to model
        session_id = self.segmenter.start_session(
            video=frame_chunk_pil,
            frame_numbers=frame_numbers,
            offload_state_to_cpu=offload_state_to_cpu,
            store_session=store_session
        )
        return {"session_id": session_id}

    def _reset_session(self, session_id: str | None = None, shm_names: dict | None = None):
        self.segmenter.reset_session(session_id)
        return {}

    def _propagate_session(self, session_id: str | None = None, shm_names: dict | None = None):
        out = self.segmenter.propagate_session(session_id)
        
        segmentation = out.segmentation  
        seg_num_frames, seg_height, seg_width = segmentation.shape

        # 1. Temporarily attach to client's segmentation memory
        assert shm_names is not None and "segmentation" in shm_names, f"No segmentation shared memory name provided. {shm_names}"
        seg_shm = shared_memory.SharedMemory(name=shm_names["segmentation"])
        try:
            seg_array = np.ndarray(self.segmentation_buffer_shape, dtype=self.segmentation_dtype, buffer=seg_shm.buf)
            # 2. Write results directly to client's memory
            seg_array[:seg_num_frames, :seg_height, :seg_width] = segmentation
        finally:
            seg_shm.close()

        return {
            "num_frames": seg_num_frames,
            "height": seg_height,
            "width": seg_width,
            "background_index": out.background_index,
            "confidences": out.confidences,
            "video_frame_indices": out.video_frame_indices,
        }

    def _add_prompt(self, prompt, frame_index: int = 0, session_id: str | None = None, shm_names: dict | None = None):
        self.segmenter.add_prompt(prompt, frame_index, session_id)
        return {}

    def _close_session(self, session_id: str | None = None, shm_names: dict | None = None):
        self.segmenter.close_session(session_id)
        return {}

    def _handle_request(self, request: dict):
        action = request.get("action")
        params = request.get("params", {})
        
        dispatch = {
            "start_session": self._start_session,
            "reset_session": self._reset_session,
            "add_prompt": self._add_prompt,
            "propagate_session": self._propagate_session,
            "close_session": self._close_session,
        }

        if action in dispatch:
            return dispatch[action](**params)
        else:
            raise ValueError(f"Invalid request: {request}")

    def _handle_connection(self, conn):
        """Executed by a thread in the ThreadPoolExecutor"""
        try:
            initial_request = conn.recv()
            if initial_request.get("action") == "acquire_lock":
                conn.send({"status": "success", "message": "Acquired lock", "data": None})
                process_request = conn.recv()
                res = self._handle_request(process_request)
                conn.send({"status": "success", "message": "Processed request", "data": res})
            else:
                conn.send({"status": "error", "message": "Invalid initial request", "data": None})
        except Exception as e:
            conn.send({"status": "error", "message": f"Server Error: {str(e)}", "data": None})
        finally:
            conn.close()
        
    def run(self):
        try:
            with Listener(self.address) as listener:
                print(f"[SAM3 SERVER] Listening on {self.address}")
                while True:
                    conn = listener.accept()
                    # Dispatch to thread pool instead of blocking
                    self.executor.submit(self._handle_connection, conn)
        except KeyboardInterrupt:
            print("[SAM3 SERVER] Interrupted. Shutting down...")
        finally:
            self.executor.shutdown(wait=True)

class _ClientSHMContext:
    """Helper to manage unique shared memory blocks for a single concurrent session."""
    def __init__(self, client_id, max_frames, width, height, f_dtype, s_dtype):
        self.task_id = str(uuid.uuid4())
        self.frame_shm_name = f"SAM3_F_{client_id}_{self.task_id}"
        self.seg_shm_name = f"SAM3_S_{client_id}_{self.task_id}"

        f_bytes, self.f_shape = _get_buffer_specs(max_frames, width, height, 3, f_dtype)
        s_bytes, self.s_shape = _get_buffer_specs(max_frames, width, height, 1, s_dtype)

        self.frame_shm = shared_memory.SharedMemory(create=True, size=f_bytes, name=self.frame_shm_name)
        self.seg_shm = shared_memory.SharedMemory(create=True, size=s_bytes, name=self.seg_shm_name)

        self.frame_array = np.ndarray(self.f_shape, dtype=f_dtype, buffer=self.frame_shm.buf)
        self.seg_array = np.ndarray(self.s_shape, dtype=s_dtype, buffer=self.seg_shm.buf)

        self.payload = {"frame": self.frame_shm_name, "segmentation": self.seg_shm_name}

    def cleanup(self):
        for shm in [self.frame_shm, self.seg_shm]:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass


class AsyncSAM3HarnessClient:
    def __init__(
        self,
        max_num_frames: int,
        max_frame_width: int,
        max_frame_height: int,
        frame_dtype: np.dtype,
        segmentation_dtype: np.dtype,
        address: tuple[str, int] | None = None,
    ):
        self.address = address if address is not None else DEFAULT_SAM3_SERVER_ADDRESS
        self.max_num_frames = max_num_frames
        self.max_frame_width = max_frame_width
        self.max_frame_height = max_frame_height
        self.frame_dtype = frame_dtype
        self.segmentation_dtype = segmentation_dtype

        self.client_id = str(uuid.uuid4())
        self.active_contexts: dict[str, _ClientSHMContext] = {}

    async def _async_request(self, request_payload: dict):
        """Wraps the synchronous socket connection in an async thread to prevent blocking."""
        def _sync_call():
            with Client(self.address) as conn:
                conn.send({"action": "acquire_lock"})
                resp = conn.recv()
                if resp.get("status") != "success":
                    raise RuntimeError(f"Lock failed: {resp.get('message')}")
                conn.send(request_payload)
                return conn.recv()
        
        return await asyncio.to_thread(_sync_call)

    async def start_session(self, video, frame_numbers: list[int] | None = None, offload_state_to_cpu: bool | None = None, store_session: bool = False) -> str:
        # Prepare video frames based on type
        if isinstance(video, Path):
            if video.is_dir():
                valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
                img_paths = sorted([p for p in video.iterdir() if p.suffix.lower() in valid_exts])
                video_frames = [np.array(Image.open(p).convert("RGB")) for p in img_paths]
            else:
                frames_arr = iio.imread(video)
                video_frames = [frames_arr[i] for i in range(frames_arr.shape[0])]
        elif isinstance(video, np.ndarray):
            video_frames = [video[i] for i in range(video.shape[0])]
        else:
            video_frames = [np.array(img.convert("RGB")) for img in video]

        num_frames = len(video_frames)
        if num_frames == 0: raise ValueError("Video contains no frames.")
        height, width = video_frames[0].shape[:2]

        # Allocate unique memory blocks for this specific concurrent session
        ctx = _ClientSHMContext(self.client_id, self.max_num_frames, self.max_frame_width, self.max_frame_height, self.frame_dtype, self.segmentation_dtype)

        # Overwrite the temporary shared memory array with new frames
        for i, frame in enumerate(video_frames):
            ctx.frame_array[i, :height, :width] = frame

        request = {
            "action": "start_session",
            "params": {
                "num_frames": num_frames, "height": height, "width": width,
                "frame_numbers": frame_numbers, "offload_state_to_cpu": offload_state_to_cpu,
                "store_session": store_session, "shm_names": ctx.payload
            }
        }
        
        resp = await self._async_request(request)
        if resp.get("status") != "success":
            ctx.cleanup()
            raise RuntimeError(f"Server error: {resp.get('message')}")

        session_id = resp["data"]["session_id"]
        self.active_contexts[session_id] = ctx # Store context so propagate/close can access it
        return session_id

    async def add_prompt(self, prompt, frame_index: int = 0, session_id: str | None = None):
        assert session_id in self.active_contexts, "No active session found for session_id."
        ctx = self.active_contexts[session_id]
        resp = await self._async_request({
            "action": "add_prompt",
            "params": {"prompt": prompt, "frame_index": frame_index, "session_id": session_id, "shm_names": ctx.payload}
        })
        if resp.get("status") != "success": raise RuntimeError(f"Server error: {resp.get('message')}")

    async def propagate_session(self, session_id: str | None = None):
        assert session_id in self.active_contexts, "No active session found for session_id."
        ctx = self.active_contexts[session_id]
        resp = await self._async_request({
            "action": "propagate_session",
            "params": {"session_id": session_id, "shm_names": ctx.payload}
        })
        if resp.get("status") != "success": raise RuntimeError(f"Server error: {resp.get('message')}")

        data = resp["data"]
        num_frames, height, width = data["num_frames"], data["height"], data["width"]

        # Safe extraction: Copy out of the session-specific shared memory
        segmentation_copy = ctx.seg_array[:num_frames, :height, :width].copy()

        return SAM3VideoOutput(
            segmentation=segmentation_copy,
            confidences=data["confidences"],
            video_frame_indices=data["video_frame_indices"]
        )

    async def close_session(self, session_id: str | None = None):
        if session_id not in self.active_contexts:
            return    

        ctx = self.active_contexts.pop(session_id)

        resp = await self._async_request({
            "action": "close_session",
            "params": {"session_id": session_id, "shm_names": ctx.payload}
        })
        
        # Always destroy memory blocks when session ends to prevent leaks
        ctx.cleanup()
        if resp.get("status") != "success": raise RuntimeError(f"Server error: {resp.get('message')}")

    async def process_batch(self, prompt, video, frame_numbers=None, offload_state_to_cpu=None):
        """Helper to orchestrate a full request cycle synchronously-looking but asynchronously executed."""
        session_id = await self.start_session(video, frame_numbers, offload_state_to_cpu, store_session=False)
        try:
            await self.add_prompt(prompt, frame_index=0, session_id=session_id)
            out = await self.propagate_session(session_id)
            return out
        finally:
            await self.close_session(session_id)