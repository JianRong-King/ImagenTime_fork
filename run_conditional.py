import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import torch.multiprocessing
import logging
import torch.nn.functional as F

from utils.loggers import NeptuneLogger, PrintLogger, CompositeLogger
from models.model import ImagenTime
from models.sampler import DiffusionProcess
from utils.utils import save_checkpoint, restore_state, create_model_name_and_dir, print_model_params, \
    log_config_and_tags, get_x_and_mask
from utils.utils_data import gen_dataloader
from utils.utils_args import parse_args_cond

torch.multiprocessing.set_sharing_strategy('file_system')


def main(args):
    # model name and directory
    name = create_model_name_and_dir(args)

        # After the training loop, save the final model (and optimizer if needed)

    def save_final_model(model, optimizer, log_dir):
        # Ensure the directory exists
        os.makedirs(log_dir, exist_ok=True)  # This will create the directory if it doesn't exist
        
        # Use os.path.join to handle path separation correctly
        final_model_path = os.path.join(log_dir, 'final_model.pth')
        final_optimizer_path = os.path.join(log_dir, 'final_optimizer.pth')

        # Save the model state_dict
        try:
            torch.save(model.state_dict(), final_model_path)
            print(f"Final model saved to {final_model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

        # Optionally, save the optimizer state_dict
        try:
            torch.save(optimizer.state_dict(), final_optimizer_path)
            print(f"Final optimizer state saved to {final_optimizer_path}")
        except Exception as e:
            print(f"Error saving optimizer state: {e}")

    # log args
    logging.info(args)

    # set-up neptune logger. switch to your desired logger
    with CompositeLogger([NeptuneLogger()]) if args.neptune \
            else PrintLogger() as logger:

        # log config and tags
        log_config_and_tags(args, logger, name)

        # --- set-up data and device ---
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader = gen_dataloader(args)
        logging.info(args.dataset + ' dataset is ready.')

        model = ImagenTime(args=args, device=args.device).to(args.device)

        # optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        state = dict(model=model, epoch=0)
        init_epoch = 0

        # restore checkpoint
        if args.resume:
            ema_model = model.model_ema if args.ema else None # load ema model if available
            init_epoch = restore_state(args, state, ema_model=ema_model)

        # print model parameters
        print_model_params(logger, model)

        # --- save initial model ---
        save_final_model(model, optimizer, args.log_dir)

        sys.exit()

        # --- train model ---
        logging.info(f"Continuing training loop from epoch {init_epoch}.")
        best_score = float('inf')  # marginal score for long-range metrics, dice score for short-range metrics
        for epoch in range(init_epoch, args.epochs):
            model.train()
            model.epoch = epoch
            logger.log_name_params('train/epoch', epoch)

            # --- train loop ---
            for i, data in enumerate(train_loader, 1):
                mask_ts, x_ts = get_x_and_mask(args, data)

                # transform to image
                x_ts_img = model.ts_to_img(x_ts)
                # pad mask with 1
                mask_ts_img = model.ts_to_img(mask_ts,pad_val=1)
                optimizer.zero_grad()
                loss = model.loss_fn_impute(x_ts_img, mask_ts_img)
                if len(loss) == 2:
                    loss, to_log = loss
                    for key, value in to_log.items():
                        logger.log(f'train/{key}', value, epoch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                model.on_train_batch_end()

                break

            # --- evaluation loop ---
            if epoch % args.logging_iter == 0:
                mse = 0
                mae = 0
                model.eval()
                with torch.no_grad():
                    with model.ema_scope():
                        process = DiffusionProcess(args, model.net,
                                                   (args.input_channels, args.img_resolution, args.img_resolution))
                        for idx, data in enumerate(test_loader, 1):
                            mask_ts, x_ts = get_x_and_mask(args, data)

                            # transform to image
                            x_ts_img = model.ts_to_img(x_ts)
                            mask_ts_img = model.ts_to_img(mask_ts, pad_val=1)

                            # sample from the model
                            # and impute, both interpolation and extrapolation are similar just the mask is different
                            x_img_sampled = process.interpolate(x_ts_img, mask_ts_img).to(x_ts_img.device)
                            x_ts_sampled = model.img_to_ts(x_img_sampled)

                            # task evaluation
                            mse_mean = F.mse_loss(x_ts[mask_ts == 0].to(x_ts.device), x_ts_sampled[mask_ts == 0])
                            mae_mean = F.l1_loss(x_ts[mask_ts == 0].to(x_ts.device), x_ts_sampled[mask_ts == 0])
                            mse += mse_mean.item()
                            mae += mae_mean.item()

                            break

                scores = {'mse': mse / (idx + 1), 'mae': mae / (idx + 1)}
                for key, value in scores.items():
                    logger.log(f'test/{key}', value, epoch)

                # --- save checkpoint ---
                curr_score = scores['mse']
                if curr_score < best_score:
                    best_score = curr_score
                    ema_model = model.model_ema if args.ema else None
                    save_checkpoint(args.log_dir, state, epoch, ema_model)
            break
        
        save_final_model(model, optimizer, args.log_dir)
        logging.info("Training is complete")




if __name__ == '__main__':
    args = parse_args_cond()  # parse unconditional generation specific args
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)