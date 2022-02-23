import sys
from utils.BuildGraph import *
from utils.Settings import *
from model.BaseNet import *
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils.Run import *
from args.BasicSet import *
import time


def main(renew_dataset, renew_graph, eval_times, task_name, device,
         batch_size, lr, weight_decay, num_epochs, **kwargs):
    start = time.time()
    result_pd = pd.DataFrame(columns=(kwargs['label_name_list'] + ['group']) * 3)
    print("Molecule graph generation is complete !")
    for time_id in range(eval_times):
        set_random_seed(2022 + time_id)
        if renew_graph:
            if renew_dataset:
                random_split(**kwargs)
            build_and_save(**kwargs)
        train_set, val_set, test_set = load_graph(**kwargs)
        print('***************************************************************************************************')
        print('{}, {}/{} time on device: {}'.format(task_name, time_id + 1, eval_times, device))
        print('***************************************************************************************************')
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=collate_molgraphs)
        val_loader = DataLoader(dataset=val_set,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=collate_molgraphs)
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=batch_size,
                                 collate_fn=collate_molgraphs)
        model = MPNN(**kwargs)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        stopper = EarlyStopping(task_name=task_name, **kwargs)
        model.to(device)

        for epoch in range(num_epochs):
            train_epoch(device=device, epoch=epoch, model=model,
                        num_epochs=num_epochs, data_loader=train_loader,
                        optimizer=optimizer, **kwargs)
            val_result = eval_epoch(device, model, val_loader, **kwargs)
            val_score = np.mean(val_result)
            early_stop = stopper.step(val_score, model)
            print('epoch {:d}/{:d}, validation {:.4f}, best validation {:.4f}'.format(
                epoch + 1, num_epochs,
                val_score, stopper.best_score) + ' validation result:', val_result)
            if early_stop:
                break

        stopper.load_checkpoint(model, device)
        test_score = eval_epoch(device, model, test_loader, **kwargs)
        train_score = eval_epoch(device, model, train_loader, **kwargs)
        val_score = eval_epoch(device, model, val_loader, **kwargs)

        result = train_score + ['training'] + val_score + ['valid'] + test_score + ['test']
        result_pd.loc[time_id] = result
        print('********************************{}, {}_times_result*******************************'.format(
            task_name,
            time_id + 1))
        print("training_result:", train_score)
        print("val_result:", val_score)
        print("test_result:", test_score)
        torch.cuda.empty_cache()

    elapsed = (time.time() - start)
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print("Time used:", "{:d}:{:d}:{:d}".format(int(h), int(m), int(s)))
    return result_pd


if __name__ == "__main__":
    settings = settings_RGCN
    settings["patience"] = 100
    settings["epoches"] = 1000
    check_settings_rgcn(settings)
    result = main(**settings)
    new_result = print_log(settings, result)[0]
    print(new_result)
