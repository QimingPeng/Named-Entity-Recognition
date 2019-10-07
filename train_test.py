import torch
import torch.optim as optim
import numpy as np
import logging
import os
from utils import WarmupLinearSchedule, entity_recover, metric

def train_eval(train_dataloader, dev_dataloader, dev_entity_list, model, id2tag, 
                ckpt_path, train_steps, check_step, eval_step, lr, warmup_steps, cv_i):
    """
        训练模型
    """
    ckpt_path = os.path.join(ckpt_path, "pytorch_model_{}.pkl".format(cv_i))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(model.named_parameters)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps, train_steps)

    model.train()
    best_step = 0
    best_F1 = 0
    train_loss = 0
    global_step = 0
    bar = range(train_steps)
    for step in bar:
        batch = next(train_dataloader)
        batch = tuple(t.to(device) for t in batch)
        batch_tensor, batch_sent_len, batch_tags = batch

        loss = model(batch_tensor, batch_sent_len, batch_tags)
        optimizer.zero_grad()

        loss.backward()
        train_loss += loss.item()
        
        optimizer.step()
        scheduler.step()
        if (step+1) % check_step == 0:
            logging.info("Step: {}, Train_batch_loss: {}".format(step+1, train_loss/check_step))
            train_loss = 0
        if (step + 1) % eval_step == 0:
            model.eval()
            output_tag_ids = []
            with torch.no_grad():
                for dev_step, (dev_tensor, dev_sent_len, dev_tags) in enumerate(dev_dataloader):
                    dev_tensor = dev_tensor.to(device)
                    dev_sent_len = dev_sent_len.to(device)
                    dev_tags = dev_tags.to(device)

                    tag_seq = model.sequence_decode(dev_tensor, dev_sent_len)
                    output_tag_ids = output_tag_ids + tag_seq
            model.train()
            output_tag = entity_recover(output_tag_ids, id2tag)
            P, R, F1 = metric(output_tag, dev_entity_list)
            
            if F1 > best_F1:
                best_step = step + 1
                best_F1 = F1
                torch.save(model.state_dict(), ckpt_path)

            logging.info("P: {:.4f}, R: {:.4f}, F1: {:.4f}, Best_f1: {:.4f}\n".format(P, R, F1, best_F1))

        if step + 1 - best_step > 5000:
            logging.info("Early stopped at Step: {}, Best_dev_f1: {}\n".format(global_step, best_F1))
            break
    return best_F1


def test(test_dataloader, test_entity_list, id2tag, model):
    """
        测试模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    output_tag_ids = []
    with torch.no_grad():
        for _, (batch_tensor, batch_sent_len, batch_tags) in enumerate(test_dataloader):
            batch_tensor = batch_tensor.to(device)
            batch_sent_len = batch_sent_len.to(device)
            batch_tags = batch_tags.to(device)
            tag_seq = model.sequence_decode(batch_tensor, batch_sent_len)
            output_tag_ids = output_tag_ids + tag_seq
    output_tag = entity_recover(output_tag_ids, id2tag)
    P, R, F1 = metric(output_tag, test_entity_list)
    
    return output_tag_ids, P, R, F1