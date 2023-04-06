import openai
from openai.upload_progress import BufferReader
import logging
import datetime
import time


openai.api_key = "geturownkey"
seed = 'who'
base_model = 'ada'
api_type = 'open_ai'
training_file = 'training.jsonl'


def setup_logger(name):
    logging.basicConfig(level=logging.DEBUG)
    return logging.getLogger(name)

def main():
    fine_tuning_job = create_model(training_file, model=base_model)
    wait_for_fine_tuning(fine_tuning_job)
    tuned_model = stream_fine_tuning(fine_tuning_job)
    print(tuned_model)

def upload_file(file = None, content = None):

        if (file is None) == (content is None):
            raise ValueError("Exactly one of `file` or `content` must be provided")

        if content is None:
            assert file is not None
            with open(file, "rb") as f:
                content = f.read()

        buffer_reader = BufferReader(content, desc="Upload progress")
        resp = openai.File.create(
            file=buffer_reader,
            purpose="fine-tune",
            user_provided_filename=file,
        )
        logger.info(
            "Uploaded file from {file}: {id}".format(
                file=file, id=resp["id"]
            )
        )
        return resp["id"]

def create_model(training_file, validation_file=None, **kwargs):
    create_args = {"training_file": upload_file(training_file)}
    if validation_file:
        create_args["validation_file"] = upload_file(validation_file)

    for key in kwargs.keys():
        if kwargs[key] != None and key in (
            "model",
            "suffix",
            "n_epochs",
            "batch_size",
            "learning_rate_multiplier",
            "prompt_loss_weight",
            "compute_classification_metrics",
            "classification_n_classes",
            "classification_positive_class",
            "classification_betas",
        ):
            create_args[key] = kwargs[key]

    resp = openai.FineTune.create(**create_args)

    logger.info(f"Created fine-tuning job: {resp['id']}")
    logger.debug(resp)

    return resp["id"]

def wait_for_fine_tuning(job_id, wait_interval=5):
    status = 'pending'
    while status == 'pending':
        time.sleep(5)
        resp = openai.FineTune.retrieve(id=job_id)
        status = resp['status']
        logger.debug(resp)

def stream_fine_tuning(job_id):
    events = openai.FineTune.stream_events(job_id)
    try:
        for event in events:
            logger.info(
                "[%s] %s"
                % (
                    datetime.datetime.fromtimestamp(event["created_at"]),
                    event["message"],
                )
            )
    except Exception as e:
        logger.warning(f'Experienced the following exception while waiting for fine tuning to finish. Will retry: {e}')
        events = None
        while events == None:
            try:
                resp = openai.FineTune.retrieve(id=job_id)
                if resp["status"] in ['pending', 'running']:
                    events = openai.FineTune.stream_events(job_id)
                else:
                    break
            except Exception as e:
                logger.warning(f'Experienced the following exception while retrying connection to fine tuning job. Will retry: {e}')

    resp = openai.FineTune.retrieve(id=job_id)
    logger.debug(resp)

    if resp["status"] == "succeeded":
        logger.info(f'Fine tuning model creation has succeeded! The resulting model is: {resp["fine_tuned_model"]}')
        return resp["fine_tuned_model"]
    elif resp["status"] == "failed":
        logger.error(f'Fine tuning model creation has failed!')
        return None

if __name__ == "__main__":
    logger = setup_logger('TwitchGPT')
    main()