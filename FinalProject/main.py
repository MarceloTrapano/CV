from utils import *
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

def main():
    dataset_dir = './Fruits Classification/all'
    batch_size = 32
    num_workers = 4
    
    transform = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.ToTensor(),
    ])
    
    lightning_callback = [ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='best-checkpoint'
    )]
    csv_logger = CSVLogger("logs", name="test")
    wandb_logger = WandbLogger(project="pytorch-lightning")
    loggers = [csv_logger, wandb_logger]

    model = LightingModel()
    data = LightingData(dataset_dir, batch_size, num_workers, transform=transform)
    
    trainer = L.Trainer(
        max_epochs=20,
        logger=loggers,
        callbacks=lightning_callback,
        deterministic=True,
        val_check_interval=1.0,
    )
    
    trainer.fit(model, data)
    
    # check on test dataset
    trainer.test(model, data.test_dataloader())
    # show example predictions
    
    
    
    
if __name__ == "__main__":
    main()