# Recommendation_System_YelpData
Data : https://drive.google.com/drive/folders/1SIlY40owpVcGXJw3xeXk76afCwtSUx11

Collaborative filtering is a technique widely used in recommendation systems. At a high level, collaborative filtering recommends items to users based on the preferences and behaviors of similar users. There are mainly two types of collaborative filtering:

User-based Collaborative Filtering: This method finds users that are similar to the target user and recommends items those similar users have liked.
Item-based Collaborative Filtering: This method finds items that are similar to the items the target user has liked and recommends those similar items.
Model-based Collaborative Filtering is a sub-category and an evolution of the original collaborative filtering approach. Instead of relying solely on user-item interactions as in the classic methods, it uses machine learning algorithms to predict a user's interest in an item.

Advantages:
1. Scalability: Model-based methods can handle larger datasets better than memory-based methods (classic user/item-based methods).
2. Handling Sparsity: User-item interaction matrices in real-world scenarios are often very sparse. Model-based methods can work well even with this sparsity.
3. Latent Feature Discovery: By compressing user-item matrices or using deep learning architectures, latent (not directly observable) features can be discovered which might be helpful for making better recommendations.
