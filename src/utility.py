import os
from huggingface_hub import notebook_login
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()


from src.custom_logger import setup_logger
#from src import setup_logger
logger=setup_logger("utility")


login(token=os.getenv("huggingface_token"))

def push_hub(model,trainer,base_model,model_save_name):

    try:
        hf_name = 'KaushiGihan' # your hf username or org name
        model_id = hf_name + "/" + base_model + model_save_name 
        model.push_to_hub(model_id)
        trainer.push_to_hub(model_id)
        
        logger.info(f"Model push to {model_id}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return None