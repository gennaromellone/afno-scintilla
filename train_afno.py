from models.afno_modulus import AFNOTrainer

if __name__ == "__main__":
    trainer = AFNOTrainer("/home/gmellone/afno-scintilla/configs/config.yaml")
    trainer.train()