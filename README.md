# GIN from scratch

A two-layer Graph Isomorphism Network. Sum aggregation and the learnable epsilon
term are implemented directly in PyTorch; `torch_geometric.nn.GINConv` is not used.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Quick test

```bash
python3 gin.py --synthetic --epochs 50
```

## OGB experiment

```bash
python3 gin.py --dataset ogbn-arxiv --epochs 100
```

GIN uses a neighbor **sum**, while GraphSAGE uses a neighbor **mean** and GCN uses
degree-normalized messages.
