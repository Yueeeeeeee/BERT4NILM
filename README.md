# BERT4NILM

PyTorch Implementation of BERT4NILM: A Bidirectional Transformer Model for Non-Intrusive Load Monitoring


## Data

The csv datasets could be downloaded here: [REDD](http://redd.csail.mit.edu/) and [UK-DALE](https://jack-kelly.com/data/)

We took the liberty of modifying certain appliance names to 'dishwasher', 'fridge', 'microwave', 'washing_machine' and 'kettle' in the 'labels.dat' file, see data folder


## Training

This is the PyTorch implementation of BERT4NILM, a bidirectional encoder representations from rransformers for energy disaggregation, in this repository we provide the BERT4NILM model as well as data functions for low frequency REDD dataset / UK Dale dataset, run following command to train an initial model, hyper-parameters (as well as appliances) could be tuned in utils.py, test will run after training ends:

```bash
python train.py
```

The trained model state dict will be saved under 'experiments/dataset-name/best_acc_model.pth'


## Performance

Our models are trained 100 / 20 epochs repspectively for appliances from REDD and UK-DALE dataset, all other parameters could be found in 'train.py' and 'utils.py'

### REDD

<img src=redd.png width=500>

### UK-DALE

<img src=uk-dale.png width=500>


## Citing 
Please cite the following paper if you use our methods in your research:
```
@inproceedings{yue2020bert4nilm,
  title={BERT4NILM: A Bidirectional Transformer Model for Non-Intrusive Load Monitoring},
  author={Yue, Zhenrui and Witzig, Camilo Requena and Jorde, Daniel and Jacobsen, Hans-Arno},
  booktitle={Proceedings of the 5th International Workshop on Non-Intrusive Load Monitoring},
  pages={89--93},
  year={2020}
}
```


## Acknowledgement

During the implementation we base our code mostly on the [BERT-pytorch](https://github.com/codertimo/BERT-pytorch) by Junseong Kim, we are also inspired by the [BERT4Rec](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch) implementation by Jaewon Chung and [Transformers](https://github.com/huggingface/transformers) from Hugging Face. Many thanks to these authors for their great work!
