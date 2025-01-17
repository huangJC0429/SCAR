---GCN---
python baseline_AU-LS.py --dataset cora --model GCN --epochs 500 --hidden 16  --lr 0.01 --weight_decay 5e-3 --dropout 0.5 --epsilon -0.5 --gamma 0.5 --LC 20
python baseline_AU-LS.py --dataset cora --model GCN --epochs 500 --hidden 16  --lr 0.01 --weight_decay 5e-3 --dropout 0.5 --epsilon -0.3 --gamma 0.5 --LC 40
python baseline_AU-LS.py --dataset cora --model GCN --epochs 500 --hidden 16  --lr 0.01 --weight_decay 5e-3 --dropout 0.5 --epsilon -0.3 --gamma 0.5 --LC 60


python baseline_AU-LS.py --dataset citeseer --model GCN --epochs 500 --hidden 16  --lr 0.01 --weight_decay 5e-3 --dropout 0.5 --epsilon -0.3 --gamma 0.5 --LC 20
python baseline_AU-LS.py --dataset citeseer --model GCN --epochs 500 --hidden 16  --lr 0.01 --weight_decay 5e-3 --dropout 0.5 --epsilon -0.3 --gamma 0.5 --LC 40
python baseline_AU-LS.py --dataset citeseer --model GCN --epochs 500 --hidden 16  --lr 0.01 --weight_decay 5e-3 --dropout 0.5 --epsilon -0.3 --gamma 0.5 --LC 60

python baseline_AU-LS.py --dataset pubmed --model GCN --epochs 500 --hidden 16  --lr 0.01 --weight_decay 5e-3 --dropout 0.5 --epsilon -0.5 --gamma 0.9 --LC 20
python baseline_AU-LS.py --dataset pubmed --model GCN --epochs 500 --hidden 16  --lr 0.01 --weight_decay 5e-3 --dropout 0.5 --epsilon -0.5 --gamma 0.9 --LC 40
python baseline_AU-LS.py --dataset pubmed --model GCN --epochs 500 --hidden 16  --lr 0.01 --weight_decay 5e-3 --dropout 0.5 --epsilon -0.5 --gamma 0.9 --LC 60

python baseline_AU-LS.py --dataset cora_full --model GCN --epochs 500 --hidden 16  --lr 0.01 --weight_decay 5e-4 --dropout 0.5 --epsilon -0.5 --gamma 0.5 --LC 20
python baseline_AU-LS.py --dataset cora_full --model GCN --epochs 500 --hidden 16  --lr 0.01 --weight_decay 5e-4 --dropout 0.5 --epsilon -0.5 --gamma 0.5 --LC 40
python baseline_AU-LS.py --dataset cora_full --model GCN --epochs 500 --hidden 16  --lr 0.01 --weight_decay 5e-4 --dropout 0.5 --epsilon -0.5 --gamma 0.5 --LC 60



---GAT---

python baseline_AU-LS.py --dataset cora --model GAT --epochs 500 --hidden 8  --lr 0.0005 --weight_decay 5e-3 --dropout 0.6 --epsilon -0.5 --gamma 0.5 --LC 20
python baseline_AU-LS.py --dataset cora --model GAT --epochs 500 --hidden 8  --lr 0.0005 --weight_decay 5e-3 --dropout 0.6 --epsilon -0.5 --gamma 0.5 --LC 40
python baseline_AU-LS.py --dataset cora --model GAT --epochs 500 --hidden 8  --lr 0.0005 --weight_decay 5e-3 --dropout 0.6 --epsilon -0.5 --gamma 0.5 --LC 60

python baseline_AU-LS.py --dataset citeseer --model GAT --epochs 500 --hidden 8  --lr 0.0005 --weight_decay 5e-3 --dropout 0.6 --epsilon -0.3 --gamma 0.5 --LC 20
python baseline_AU-LS.py --dataset citeseer --model GAT --epochs 500 --hidden 8  --lr 0.0005 --weight_decay 5e-3 --dropout 0.6 --epsilon -0.3 --gamma 0.5 --LC 40
python baseline_AU-LS.py --dataset citeseer --model GAT --epochs 500 --hidden 8  --lr 0.0005 --weight_decay 5e-3 --dropout 0.6 --epsilon -0.3 --gamma 0.5 --LC 60

python baseline_AU-LS.py --dataset pubmed --model GAT --epochs 500 --hidden 8  --lr 0.0005 --weight_decay 5e-3 --dropout 0.6 --epsilon -0.3 --gamma 0.5 --LC 20
python baseline_AU-LS.py --dataset pubmed --model GAT --epochs 500 --hidden 8  --lr 0.0005 --weight_decay 5e-3 --dropout 0.6 --epsilon -0.3 --gamma 0.5 --LC 40
python baseline_AU-LS.py --dataset pubmed --model GAT --epochs 500 --hidden 8  --lr 0.0005 --weight_decay 5e-3 --dropout 0.6 --epsilon -0.3 --gamma 0.5 --LC 60

python baseline_AU-LS.py --dataset cora_full --model GAT --epochs 2000 --hidden 8  --lr 0.0005 --weight_decay 5e-4 --dropout 0.6 --epsilon -0.3 --gamma 0.5 --LC 20
python baseline_AU-LS.py --dataset cora_full --model GAT --epochs 2000 --hidden 8  --lr 0.0005 --weight_decay 5e-4 --dropout 0.6 --epsilon -0.3 --gamma 0.5 --LC 40
python baseline_AU-LS.py --dataset cora_full --model GAT --epochs 2000 --hidden 8  --lr 0.0005 --weight_decay 5e-4 --dropout 0.6 --epsilon -0.3 --gamma 0.5 --LC 60



