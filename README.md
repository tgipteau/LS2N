# FISO (Forêt isométrique)

Les scripts présents dans ce dépôt sont des outils de simulation d'évolution de forêt.<br>
Ces évolutions sont issues de modèles développés et étudiés par l'équipe VELO du LS2N à Nantes, dans 
le cadre du projet TOUNDRA.<br>

<ul>
    <li><a href="https://pagesperso.ls2n.fr/~cantin-g/toundra.html">Plus d'informations sur le projet TOUNDRA</a></li>
    <li><a href="https://velo.pythonanywhere.com/">L'équipe VELO</a></li>
    <li><a href="https://hal.science/hal-04069502v1">On the degradation of forest ecosystems by extreme events: Statistical Model Checking of a hybrid model</a></li>
</ul>

### Les modèles

On distingue deux modèles :
<ul>
    <li> Un modèle dit "à deux équations", qui modélise l'évolution de la densité de population d'une espèce
    d'arbre et de ses graines. On ajoute à ce processus déterministe (EDP) un dérèglement stochastique par 
    l'apparition de feux de forêt aléatoires.</li>
    <li> Un modèle dit "à quatre équations", qui ajoute au modèle précédent une deuxième espèce d'arbres et ses graines.
    On ajoute alors un facteur de compétition inter-espèces, ainsi qu'un effet spatial (préférence pour le nord ou
    le sud en fonction de l'espèce). </li>
</ul>


### Contenu de ce dépôt

<b>Fiso2</b> : utilise le modèle à quatre équations (deux espèces d'arbres). Utilise FreeFem++ (logiciel externe).<br>
Ce Fiso est destiné à une utilisation en local après installation de FreeFem++. Comprend une interface utilisateur<br>
pour lancer des simulations et les charger (<i>user-friendly</i>).<br>

<b>Fiso3</b> : utilise le modèle à deux équations (une seule espèce d'arbre). C'est un programme entièrement écrit<br>
en Python, et donc portable sur une plateforme comme PythonAnywhere. Ne comprend pas d'interface utilisateur<br>
pour configurer la simulation : la configuration se fait par l'édition d'un fichier <i>.yaml</i>.<br>

<b>Fiso4</b> : utilise le modèle à quatre équations (deux espèces d'arbres). C'est un programme entièrement écrit<br>
en Python, et donc portable sur une plateforme comme PythonAnywhere.  Ne comprend pas d'interface utilisateur<br>
pour configurer la simulation : la configuration se fait par l'édition d'un fichier <i>.yaml</i>.<br>
<br>




