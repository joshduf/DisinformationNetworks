# DisinformationNetworks
A Deep Learning Approach to Identifying Covert Disinformation Networks

Covert and organized networks can use social media platforms to launch large
scale disinformation campaigns and spread their influence undetected. Much
work has gone into detecting a fake news post on social media based on content.
This is a difficult task often requiring background knowledge of a subject.

Identifying connected social networks based on their connections is also a
well-studied problem. However, the prospect of well-organized networks
attempting to hide their existence complicates this type of analysis.

This project takes a different tack and uses a convolutional neural network
(CNN) to identify these networks based on content. It was tested on tweets
from users identified by Twitter and compiled by NBCNews as being professional
members of a Russia's fake news network the Internet Research Agency (IRA).
When compared against users who tweeted similiar content but were not members
of the IRA, it was able to predict membership with 95% accuracy. When
compared against average Twitter users geolocated in the U.S., accuracy was
higher. 20 chronological tweets from a user were neccessary to provide this
level of accuracy.

This model significantly outperformed LSTMs and other sequential networks in
both accuracy and speed. These details are in the project paper.
