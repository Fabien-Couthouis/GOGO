# AlphaZero reproduction

* GO code is available in [goban.py](./goban.py). It respects the [rules provided here](./rules.png).
* Vanilla MCTS player is available in [mctsPlayer.py](./players/mctsPlayer.py).
* AlphaZero player is available in [mctsPlayer.py](./players/mctsPlayer.py), the model can be found in [alphazeroModel.py](./players/alphazeroModel.py). Missing computational power to train it on a 19x19 goban. The player is not optimized at all and maybe still bugged.


## Requirements

```
joblib==0.14.1
scipy==1.4.1
matplotlib==3.1.2
numpy==1.16.2
tensorflow==2.1.0

```

## Usage
It is possible to simulate game using [play.py](./play.py).

```python
    goban = Goban()
    player1, player2 = AlphaZeroPlayer(
        goban._WHITE, goban.get_board_size(), "saves/test1/best/model"), MCTSPlayer(goban._BLACK, goban.get_board_size())
    play(player1, player2, goban)
```

To train the AlphaZero player, simply launch from the root of the project:

```batch
python trainMcts.py --name model_name
```



