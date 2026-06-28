The best local model is now logistic regression.
Averaged across scopes, logreg is best on accuracy and macro-F1:
accuracy: 0.356
macro-F1: 0.242
top-3: 0.617
So the added features are helping the feature-based model more than the count baselines.

But the global model is still weak.
On global, the best accuracy is still markov at 0.255; logreg is only 0.233, though it has better top-3 (0.524). This suggests subject-specific behavior matters a lot, and pooling everyone together blurs patterns.

Top-3 being decent means the model often “knows the neighborhood,” but not the exact next event.
The right answer is frequently among the top candidates, but ranking the top candidate correctly is hard. That usually points to ambiguity in ADL routines, repeated activities, class imbalance, or missing state/context.

Order-3 Markov is mostly too sparse.
markov_order3 is usually worse than order-1 Markov. With this dataset size, exact 3-event contexts do not repeat enough. A backoff/interpolated Markov model would be more defensible than plain order-3.

Decision tree is not benefiting from the extra features.
Tree accuracy is poor (0.213 average), and top-3 is also weak. It is likely overfitting small local datasets and unstable splits. I would not emphasize it as a strong model.

Federated/global models lose personalization.
Local logreg averages 0.374 accuracy across subjects, while federated logreg averages 0.264. Local Markov also beats federated Markov on average. This is strong evidence that personalization/local training matters more than simply aggregating all users.

Subject5 looks artificially strong / unstable.
Subject5 has only very small validation support, so 0.556 accuracy for logreg is not as meaningful as larger subjects. Treat it carefully in thesis discussion.