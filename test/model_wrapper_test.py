"""A testing Class for Model Wrapper"""

# external
from PIL import Image
import time
import os
import psutil
import openpyxl
import pandas as pd

# internal
from configs.model_test_config import models_dict
from core.wrapper import ModelWrapper
from utils.env_manager import *
from pipeline.model_loading_pipeline import load_model
from pipeline.model_inference import model_inference
from core.logger import get_logger

LOG = get_logger("test")


def test_model_loading():

    load_model(models_dict["model_1"])


def test_inference():
    model_inference(models_dict["model_1"])


if __name__ == "__main__":

    excel_file = "torch_model_metrics.xlsx"

    model_metrics = []

    start_inference_time = time.time()
    process = psutil.Process(os.getpid())

    mem_before = process.memory_info().rss / (1024**2)

    LOG.info("process is starting...")

    load_status = None
    inference_status = None
    load_error = None
    inference_error = None

    try:
        LOG.info("starting model loading")
        test_model_loading()
        LOG.info("loaded model successfully")
        load_status = 1

    except Exception as e:
        LOG.error(f"\n\n\t Error encountered: {e}", exc_info=True)
        load_status = 0
        load_error = str(e)

    try:
        LOG.info("starting inference test")
        test_inference()
        LOG.info("loaded model successfully")
        inference_status = 1

    except Exception as e:
        LOG.error(f"\n\n\t Error encountered: {e}", exc_info=True)
        inference_status = 0
        inference_error = str(e)

    end_inference_time = time.time()
    mem_after = process.memory_info().rss / (1024**2)

    mem_used = mem_after - mem_before

    TOTAL_TIME_TAKEN = round((end_inference_time - start_inference_time) / 60, 2)
    model_data = {
        "model_name": models_dict["model_1"]["model_name"],
        "total_time_taken(mins)": TOTAL_TIME_TAKEN,
        "memory_used": mem_used,
        "load_status": load_status,
        "load_error": load_error,
        "inference_status": inference_status,
        "inference_error": inference_error,
    }

    model_metrics.append(model_data)

    df = pd.DataFrame(model_metrics)

    if os.path.exists(excel_file):
        existing_df = pd.read_excel(excel_file)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        updated_df = df

    # Save back to Excel
    updated_df.to_excel(excel_file, index=False)

    print(f"Saved {len(df)} model entries to {excel_file}")
