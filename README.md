﻿# Project-Election
 Le client aimerait obtenir une représentation des données en arbre qui puisse donner une représentation graphique adéquate des proximités entre candidats à une éléction à partir des rangements fournis par les votants.

Contraintes: -les données de rangements sont de nature single-peaked, et ceci doit se refléter sur l'arbre résultant
        -la recherche de la solution doit se faire par recherche locale ou algorithme génétique 
        -le dataset de référence est l'experience "voter autrement" 2017-2020 : 11000 rangement, 11 candidats, uniquement les 4 candidats préférés par rangement
        -la solution doit pouvoir permettre au client de charger un dataset en mémoire et de calculer et afficher l'arbre résultant de maniere autonome
        -> le calcul de l'arbre doit se faire en un temps raisonnable pour le client.
        -les parametres d'itérations ou de temps de l'algorithme doivent être paramétrable du côté client.
        -la topologie de l'arbre ne doit pas être dégénérée: ne pas être liée à la façon dont on définit la distance

solution envisagée: 

La solution consisterait en un logiciel qui permetrrait au client de charger un dataset arbitraire et de calculer un arbre reflétant au mieux les rangements des votants, puis d'afficher cet arbre.
Language choisi: Python
Le calcul de l'arbre se fait par recherche locale ou algorithme génétique en minimisant l'aggrégation des distances de l'arbre avec chaque rangement à partir d'une heuristique de distance (à identifier). 
L'agorithme à comme proritété, issue de la fonction de distance ou d'une autre composante, de ne pas donner d'arbre dégénéré
fonctionnement
-Le programme charge les données csv en mémoire
-laisse le client décider d'un nombre d'itérations ou temps de calcul
-le programme calcule un arbre minimisant, par recherche locale ou algorithme génétique minimisant la distance globale.
-le programme affiche le graphe résultat au client.****
