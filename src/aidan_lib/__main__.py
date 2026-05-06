from pydantic import BaseModel, Field, ValidationError
from typer import Typer, Option
import typer

app = Typer()

class SAM3HarnessArgs(BaseModel):
    device: str = Field(default="cuda")
    dtype: str = Field(default="bfloat16")
    inference_device: str | None = Field(default=None)
    processing_device: str = Field(default="cpu")
    video_storage_device: str = Field(default="cpu")
    compile: bool = Field(default=True)
    warm_up: bool = Field(default=True)

class SAM3ServerArgs(BaseModel):
    max_num_frames: int = Field(..., gt=0, description="Maximum number of frames to keep in memory")
    max_frame_width: int = Field(..., gt=0)
    max_frame_height: int = Field(..., gt=0)

    frame_dtype: str = Field(default="uint8", description="Dtype of the frames")

    address: tuple[str, int] = Field(default=("localhost", 26000))
    shared_frame_memory_name: str = Field(default="sam3_frames")
    shared_segmentation_memory_name: str = Field(default="sam3_segmentations")
    
    segmenter_kwargs: SAM3HarnessArgs = Field(default_factory=SAM3HarnessArgs)

def _start_sam3_server(args: SAM3ServerArgs, use_async: bool = False):
    print("Starting SAM3 server")
    if use_async:
        from aidan_lib.models.sam3_async import SAM3HarnessServer
    else:
        from aidan_lib.models.sam3_base import SAM3HarnessServer
    server = SAM3HarnessServer(
        **args.model_dump()
    )
    server.run()

@app.callback()
def main():
    """SAM3 Utilities."""
    pass

@app.command()
def start_sam3_server(
    # Server Args
    max_num_frames: int = Option(..., help="Maximum number of frames to keep in memory"),
    max_frame_width: int = Option(...),
    max_frame_height: int = Option(...),
    frame_dtype: str = Option("uint8", help="Dtype of the frames"),
    host: str = Option("localhost", help="Server host address"),
    port: int = Option(26000, help="Server port"),
    shared_frame_memory_name: str = Option("sam3_frames"),
    shared_segmentation_memory_name: str = Option("sam3_segmentations"),
    
    # SAM3Harness Args
    device: str = Option("cuda"),
    dtype: str = Option("bfloat16"),
    inference_device: str | None = Option(None),
    processing_device: str = Option("cpu"),
    video_storage_device: str = Option("cpu"),
    compile_model: bool = Option(True, "--compile/--no-compile", help="Compile the model"),
    warm_up: bool = Option(True, "--warm-up/--no-warm-up", help="Warm up the model"),
    use_async: bool = Option(False, "--async/--sync", help="Use async client"),
):
    """
    Start the SAM3 shared-memory server.

    Example command: 
    ```bash
    uv run python -m aidan_lib start-sam3-server \
        --max-num-frames 120 \
        --max-frame-width 1920 \
        --max-frame-height 1080 \
        --host localhost \
        --port 26000 \
        --shared-frame-memory-name sam3_frames \
        --shared-segmentation-memory-name sam3_segmentations \
        --device cuda \
        --dtype bfloat16 \
        --inference-device cpu \
        --processing-device cuda \
        --video-storage-device cpu \
        --no-compile \
        --no-warm-up \
        --async
    ```
    """
    try:
        # 1. Build the nested Harness args first
        segmenter_args = SAM3HarnessArgs(
            device=device,
            dtype=dtype,
            inference_device=inference_device,
            processing_device=processing_device,
            video_storage_device=video_storage_device,
            compile=compile_model,
            warm_up=warm_up
        )

        # 2. Build the main Server args
        server_args = SAM3ServerArgs(
            max_num_frames=max_num_frames,
            max_frame_width=max_frame_width,
            max_frame_height=max_frame_height,
            frame_dtype=frame_dtype,
            address=(host, port), # Recombine into the expected tuple
            shared_frame_memory_name=shared_frame_memory_name,
            shared_segmentation_memory_name=shared_segmentation_memory_name,
            segmenter_kwargs=segmenter_args
        )
    except ValidationError as e:
        typer.echo(f"Configuration Validation Error:\n{e}", err=True)
        raise typer.Exit(code=1)
    print(f"Built server args: {server_args}")

    # 3. Fire up the server
    _start_sam3_server(server_args, use_async=use_async)

if __name__ == "__main__":
    app()