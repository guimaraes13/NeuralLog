male(alfonso).
male(andrew).
male(arthur).
male(charles).
male(christopher).
male(colin).
male(emilio).
male(james).
male(marco).
male(pierro).
male(roberto).
male(tomaso).

female(angela).
female(charlotte).
female(christine).
female(francesca).
female(gina).
female(jennifer).
female(lucia).
female(margaret).
female(maria).
female(penelope).
female(sophia).
female(victoria).

age(alfonso, 2).
age(andrew, 67).
age(angela, 5).
age(arthur, 37).
age(charlotte, 11).
age(christine, 13).
age(christopher, 59).
age(colin, 19).
age(emilio, 23).
age(francesca, 29).
age(james, 61).
age(jennifer, 3).
age(lucia, 41).
age(marco, 53).
age(maria, 3).
age(penelope, 43).
age(pierro, 71).
age(roberto, 31).
age(sophia, 47).
age(victoria, 17).

#father(andrew, james).
father(andrew, jennifer).
#father(christopher, arthur).
father(christopher, victoria).
father(james, victoria).
#father(james, colin).
#father(marco, alfonso).
father(marco, sophia).
father(pierro, angela).
#father(pierro, marco).
father(roberto, emilio).
father(roberto, lucia).

mother(christine, jennifer).
mother(francesca, angela).
mother(lucia, sophia).
mother(maria, emilio).
mother(maria, lucia).
mother(penelope, victoria).
mother(victoria, charlotte).

son(james, andrew).
son(arthur, christopher).
son(colin, james).
son(alfonso, marco).
son(marco, pierro).
son(emilio, roberto).

father(X, Y) :- son(Y, X), male(X).

parent(X, Y) :- mother(X, Y).
parent(X, Y) :- father(X, Y).

grand_mother(X, Y) :- mother(X, lucia), parent(lucia, Y).
grand_grand_father(X, Y) :- father(X, james), parent(james, victoria),
    parent(victoria, Y).

avgAgeFriends(X) :- father(X, Z), age(Z, Y), mean(Y).
ageFriends(X, Y) :- father(X, Z), age(Z, Y).

0.8660::l_1(andrew).
0.5::l_2(andrew).

0.7071::l_1(christopher).
0.7071::l_2(christopher).

0.5::l_1(james).
0.8660::l_2(james).

h_1(X, Y) :- l_1(X), l_1(Y). %% X_1 * Y_1
h_2(X, Y) :- l_2(X), l_2(Y). %% X_2 * Y_2

num(X, Y) :- h_1(X, Y).  %% (X_1 * Y_1) + (X_2 * Y_2)
num(X, Y) :- h_2(X, Y).

square_sum(X) :- l_1(X), l_1(X).  %% X_1^2 + X_2^2
square_sum(X) :- l_2(X), l_2(X).

norm(X) :- square_sum(X), square_root(X).

den(X, Y) :- norm(X), norm(Y).
inv_den(X, Y) :- den(X, Y), inverse(Y).

similarity(X, Y) :- num(X, Y), inv_den(X, Y).
