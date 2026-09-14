import os
import torch
from transformers import Trainer
from transformers.utils import logging, SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from typing import Optional, Dict, Union, Any, Tuple, List
import torch.nn as nn

logger = logging.get_logger(__name__)
TRAINING_ARGS_NAME = "training_args.bin"

class LaMedTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

            if not prediction_loss_only:
                logits = outputs.logits
                # preprocess_logits_for_metrics expects (logits, labels)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, inputs.get("labels"))
            else:
                logits = None

        labels = inputs.get("labels")
        
        return (loss, logits, labels)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if state_dict is None:
            state_dict = self.model.state_dict()

        logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))