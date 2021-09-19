# Kaggle-Titanic

## Machine learning classification competition
<p align="center"><img src="https://github.com/NickKaparinos/Kaggle-Titanic/blob/master/Plots/titanic.jpg" alt="drawing" width="600"/></p>
The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships <br /><br />





One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

## Cross validation results
After applying the appropriate preprocessing, each model was evaluated using 10-fold cross validation. Each model\`s hyperparameters were tuned using extensive grid search. Afterwards, tuned models can be ensembled together for a boost in accuracy.
<p align="center"><img src="https://github.com/NickKaparinos/Kaggle-Titanic/blob/master/Plots/cv.png" alt="drawing" width="1200"/></p>

## Test set results
Using the optimal voting model, a test set accuracy of **0.8110** was achieved, which corresponds to position **349/50092** in the leaderboard (top 0.6%).

<p align="center"><img src="https://github.com/NickKaparinos/Kaggle-Titanic/blob/master/Plots/kaggle_results1.PNG" alt="drawing" width="850"/></p>
<p align="center"><img src="https://github.com/NickKaparinos/Kaggle-Titanic/blob/master/Plots/kaggle_results2.PNG" alt="drawing" width="850"/></p>
