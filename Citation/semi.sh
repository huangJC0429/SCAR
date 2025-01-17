
---for instance w---
--CCN--
python test-GCN_ins_w.py --dataset cora --model T-GCN --epochs 2000 --hidden 64  --lr 0.015 --dropout 0.6 --lr_out 0.01 --weight_decay_out 1e-4 --a 2e-4 --b 2e-3  --LC 20
python test-GCN_ins_w.py --dataset cora --model T-GCN --epochs 2000 --hidden 64  --lr 0.015 --dropout 0.6 --lr_out 0.01 --weight_decay_out 1e-4 --a 1e-4 --b 1e-3  --LC 40
python test-GCN_ins_w.py --dataset cora --model T-GCN --epochs 2000 --hidden 64  --lr 0.015 --dropout 0.6 --lr_out 0.01 --weight_decay_out 1e-4 --a 1e-4 --b 1e-3  --LC 60

python test-GCN_ins_w.py --dataset citeseer --model T-GCN --epochs 2000 --hidden 64  --lr 0.01 --dropout 0.5 --lr_out 0.01 --weight_decay_out 4.5e-4 --a 5e-5 --b 5e-3 --LC 20
python test-GCN_ins_w.py --dataset citeseer --model T-GCN --epochs 2000 --hidden 64  --lr 0.01 --dropout 0.5 --lr_out 0.01 --weight_decay_out 5e-4 --a 5e-5 --b 5e-3 --LC 40
python test-GCN_ins_w.py --dataset citeseer --model T-GCN --epochs 2000 --hidden 64  --lr 0.01 --dropout 0.5 --lr_out 0.015 --weight_decay_out 4e-4 --a 5e-4 --b 2e-3 --LC 60


--this is only for w_norm L2---
python test-GCN_ins_w.py --dataset citeseer --model T-GCN --epochs 2000 --hidden 64  --lr 0.01 --dropout 0.5 --lr_out 0.01 --weight_decay_out 3e-5 --a 5e-6 --b 5e-4

python test-GCN_ins_w.py --dataset pubmed --model T-GCN --epochs 2000 --hidden 16  --lr 0.02 --dropout 0.5 --lr_out 0.03 --weight_decay_out 4e-4 --a 5e-8 --b 5e-6 --LC 20
python test-GCN_ins_w.py --dataset pubmed --model T-GCN --epochs 2000 --hidden 16  --lr 0.02 --dropout 0.5 --lr_out 0.03 --weight_decay_out 4e-4 --a 5e-7 --b 5e-5 --LC 40
python test-GCN_ins_w.py --dataset pubmed --model T-GCN --epochs 2000 --hidden 16  --lr 0.02 --dropout 0.5 --lr_out 0.03 --weight_decay_out 4e-4 --a 5e-7 --b 5e-5 --LC 60
                                                                                                                                                                                              

python test-GCN_ins_w.py --dataset cora_full --model T-GCN --epochs 2000 --hidden 64  --lr 0.01 --dropout 0.5 --lr_out 0.01 --weight_decay_out 2e-6 --a 3e-5 --b 5e-4  --LC 20
python test-GCN_ins_w.py --dataset cora_full --model T-GCN --epochs 2000 --hidden 64  --lr 0.01 --dropout 0.5 --lr_out 0.01 --weight_decay_out 1e-4 --a 3e-4 --b 5e-3  --LC 40
python test-GCN_ins_w.py --dataset cora_full --model T-GCN --epochs 2000 --hidden 64  --lr 0.01 --dropout 0.5 --lr_out 0.01 --weight_decay_out 2e-6 --a 3e-5 --b 5e-4  --LC 60


--GAT--
python test-GAT_ins_w.py --dataset cora --model T-GAT --epochs 2000 --hidden 8  --lr 0.01 --dropout 0.6 --lr_out 0.01 --weight_decay_out 2e-6 --a 5e-5 --b 5e-4  --LC 20


python test-GAT_ins_w.py --dataset citeseer --model T-GAT --epochs 2000 --hidden 8  --lr 0.01 --dropout 0.6 --lr_out 0.01 --weight_decay_out 3e-4 --a 5e-4 --b 5e-3  --LC 20
python test-GAT_ins_w.py --dataset citeseer --model T-GAT --epochs 2000 --hidden 8  --lr 0.01 --dropout 0.6 --lr_out 0.01 --weight_decay_out 3e-4 --a 5e-4 --b 5e-3  --LC 40
python test-GAT_ins_w.py --dataset citeseer --model T-GAT --epochs 2000 --hidden 8  --lr 0.01 --dropout 0.6 --lr_out 0.01 --weight_decay_out 3e-4 --a 5e-4 --b 5e-3  --LC 60




python test-GAT_ins_w.py --dataset pubmed --model T-GAT --epochs 2000 --hidden 8  --lr 0.01 --dropout 0.6 --lr_out 0.01 --weight_decay_out 3e-4 --a 5e-6 --b 5e-5  --LC 20
python test-GAT_ins_w.py --dataset pubmed --model T-GAT --epochs 2000 --hidden 6  --lr 0.01 --dropout 0.5 --lr_out 0.01 --weight_decay_out 3e-4 --a 5e-7 --b 5e-5  --LC 40
python test-GAT_ins_w.py --dataset pubmed --model T-GAT --epochs 2000 --hidden 8  --lr 0.01 --dropout 0.5 --lr_out 0.01 --weight_decay_out 3e-4 --a 5e-7 --b 5e-6  --LC 60



python test-GAT_ins_w.py --dataset cora_full --model T-GAT --epochs 2000 --hidden 8  --lr 0.01 --dropout 0.6 --lr_out 0.01 --weight_decay_out 2e-6 --a 3e-5 --b 5e-4  --LC 60


---Ablation Study---

python test-GCN_ins_w.py --dataset citeseer --model T-GCN --epochs 2000 --hidden 64  --lr 0.01 --dropout 0.5 --lr_out 0.01 --weight_decay_out 5e-4 --a 5e-5 --b 5e-3 --LC 20