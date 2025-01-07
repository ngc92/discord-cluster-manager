import time
from typing import Optional

import discord
import modal
from consts import ModalGPU
from discord import app_commands
from discord.ext import commands
from leaderboard_eval import cu_eval, py_eval
from modal_runner_archs import (  # noqa: F401
    run_cuda_script_a100,
    run_cuda_script_h100,
    run_cuda_script_l4,
    run_cuda_script_t4,
    run_pytorch_script_a100,
    run_pytorch_script_h100,
    run_pytorch_script_l4,
    run_pytorch_script_t4,
)
from utils import send_discord_message, setup_logging

logger = setup_logging()


class ModalCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

        self.run_modal = bot.run_group.command(
            name="modal", description="Run a script using Modal"
        )(self.run_modal)

    @app_commands.describe(
        script="The Python script file to run", gpu_type="Choose the GPU type for Modal"
    )
    @app_commands.choices(
        gpu_type=[app_commands.Choice(name=gpu.name, value=gpu.value) for gpu in ModalGPU]
    )
    async def run_modal(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
        gpu_type: app_commands.Choice[str],
        reference_script: discord.Attachment = None,
        reference_code: str = None,
    ) -> discord.Thread:
        thread = None
        try:
            if not script.filename.endswith(".py") and not script.filename.endswith(".cu"):
                await send_discord_message("Please provide a Python (.py) or CUDA (.cu) file")
                return None

            thread = await self.bot.create_thread(interaction, gpu_type.name, "Modal Job")
            queue_start_time = time.perf_counter()

            await thread.send(f"**Processing `{script.filename}` with {gpu_type.name}...**")

            script_content = (await script.read()).decode("utf-8")
            status_msg = await thread.send(
                "**Running on Modal...**\n> ⏳ Waiting for available GPU..."
            )

            script_content = (await script.read()).decode("utf-8")
            filename = "train.py" if script.filename.endswith(".py") else "train.cu"

            if reference_script is not None or reference_code is not None:
                reference_content = (
                    reference_code
                    if reference_code is not None
                    else (await reference_script.read()).decode("utf-8")
                )
                result, score = await self.trigger_modal_run(
                    script_content,
                    filename,
                    gpu_type.value,
                    reference_content,
                )
            else:
                result, score = await self.trigger_modal_run(
                    script_content, filename, gpu_type.value
                )
                queue_end_time = time.perf_counter()
                queue_time = queue_end_time - queue_start_time

                # Send metrics and results
                await thread.send(f"\n**Script size:** {len(script_content)} bytes")
                await thread.send(f"**Queue time:** {queue_time:.3f} s")
                await thread.send(f"**Execution time:** {score:.3f} s\n")
                await thread.send(f"**Modal execution result:**\n```\n{result}\n```")

            if "check_implementation failed" in result:
                await thread.send("check_implementation failed.\n")
                await status_msg.edit(content="**Running on Modal...**\n> ❌ Job failed!")
                return thread
            elif "Error" in result:
                await thread.send(f"Modal run failed. {result}\n")
                await status_msg.edit(content="**Running on Modal...**\n> ❌ Job failed!")
                return thread

            if result is not None:
                await thread.send(f"**score:{score:.9f}**\n```")

            # Update status message to show completion
            await status_msg.edit(content="**Running on Modal...**\n> ✅ Job completed!")

            return thread

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            if thread:
                # Update status message to show error
                await status_msg.edit(content="**Running on Modal...**\n> ❌ Job failed!")
                await thread.send(f"**Error:** {str(e)}")
            raise

    # TODO: needs cleaning with modal folks
    async def trigger_modal_run(  # noqa: C901
        self,
        script_content: str,
        filename: str,
        gpu_type: str,
        reference_content: Optional[str] = None,
    ) -> tuple[str, float]:
        logger.info("Attempting to trigger Modal run")

        from modal_runner import app

        try:
            print(f"Running {filename} with Modal")
            gpu_type = gpu_type.lower()
            with modal.enable_output():
                with app.run():
                    if reference_content is not None:
                        if filename.endswith(".py"):
                            from modal_runner_archs import get_pytorch_modal_runner

                            modal_runner = get_pytorch_modal_runner(gpu_type)
                            stdout, score = modal_runner.remote(
                                py_eval,
                                reference_content=reference_content,
                                submission_content=script_content,
                            )
                        else:
                            from modal_runner_archs import get_cuda_modal_runner

                            modal_runner = get_cuda_modal_runner(gpu_type)
                            stdout, score = modal_runner.remote(
                                cu_eval,
                                reference_content=reference_content,
                                submission_content=script_content,
                            )
                    else:
                        if filename.endswith(".py"):
                            from modal_runner_archs import get_pytorch_modal_runner

                            modal_runner = get_pytorch_modal_runner(gpu_type)
                            stdout, score = modal_runner.remote(script_content)
                        elif filename.endswith(".cu"):
                            from modal_runner_archs import get_cuda_modal_runner

                            modal_runner = get_cuda_modal_runner(gpu_type)
                            stdout, score = modal_runner.remote(script_content)

            return stdout, score

        except Exception as e:
            logger.error(f"Error in trigger_modal_run: {str(e)}", exc_info=True)
            return f"Error: {str(e)}", 0
