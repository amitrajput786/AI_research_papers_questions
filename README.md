# AI Research Papers Questions

This repository contains the collection of important algorithms from research papers to understand AI better.

## Contents
### Densenet
- `problem-30.py`-[Single Neuron with Backpropagation]
- `problem-41.py`-[implement simple convolutional 2d layer ]
""" output_size of kernel : 
                          output_h=(input_h+2*padding-kernel_h)//stride+1
                          output_w=(input_w+padding-kernel_w)//stride+1
         kernel slide over the input by stride , and padding added the 0's to protect the spatial dimensions and elementwise multiplication takesplace                        """

### Resnet
- `problem-25.py` - Implementation of ResNet architecture
- `problem-26.py` - [implement of single nueron by backprpophgation]
- `problem-41.py` - [implementation of simple convolution 2D layer]
- `problem-113.py` - [implement a residual block with shortcut connection]
- `problem-114,115.py`-[implementation of global avearge pooling and Batchnormalization]
"""┌─────────────────────────────────────────────────────────────────────────┐
│                      USE CASES OF BATCH NORMALIZATION                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. FASTER TRAINING                                                     │
│     ├── Allows higher learning rates                                    │
│     └── Reduces training time significantly                             │
│                                                                         │
│  2. INTERNAL COVARIATE SHIFT                                           │
│     ├── Stabilizes layer input distributions                            │
│     └── Each layer sees consistent input statistics                     │
│                                                                         │
│  3. REGULARIZATION EFFECT                                               │
│     ├── Adds noise during training (batch statistics)                   │
│     └── Slight regularization, can reduce need for dropout              │
│                                                                         │
│  4. GRADIENT FLOW                                                       │
│     ├── Prevents vanishing/exploding gradients                          │
│     └── Enables training of very deep networks                          │
│                                                                         │
│  5. WEIGHT INITIALIZATION                                               │
│     ├── Less sensitive to weight initialization                         │
│     └── Networks train well even with simple initialization             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘"""

## How to Use
```bash
# Clone the repository
git clone https://github.com/amitrajput786/AI_research_papers_questions.git

# Navigate to specific problem
cd AI_research_papers_questions/Resnet

# Run the code
python problem-25.py
```

## License

MIT License