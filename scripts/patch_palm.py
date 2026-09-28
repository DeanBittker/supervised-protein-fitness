import os
import sys

# PALM's LightAttention predictor refuses regression in three places: it will not build
# the model, its forward pass thresholds against a classification cutoff, and after
# training it looks for an optimal cutoff. None of those are properties of the network,
# which is sigmoid bounded and trains happily against a continuous target in [0, 1]
# once the labels are min-max scaled. This applies the smallest change that lets it run
# and leaves the architecture and the loss alone.
#
#   python3 scripts/patch_palm.py ~/PALM

palm_root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/PALM")
target = f"{palm_root}/src/model/predictors.py"

patches = [
    (
        "build the model in regression mode",
        '''                if cfg.predictor.model_type == "regression":
                    raise NotImplementedError("Regression not implemented for LightAttention")
                elif cfg.predictor.model_type == "classification_binary":
                    # Lazy load (once we know the embedding dimension)
                    self.model = None''',
        '''                if cfg.predictor.model_type in ("regression", "classification_binary"):
                    # Lazy load (once we know the embedding dimension)
                    self.model = None''',
    ),
    (
        "return the bounded output rather than thresholding it",
        '''        if self.model_type == "classification_binary":
            with torch.inference_mode():
                predicted_probabilities = self.model.forward(padded_embeddings, mask)
                predicted_probabilities = self.model.convert_to_numpy(predicted_probabilities, mask)
            predictions = (
                predicted_probabilities >= self.optimal_cutoff if self.optimal_cutoff else None
            )
        else:
            raise NotImplementedError''',
        '''        with torch.inference_mode():
            predicted_probabilities = self.model.forward(padded_embeddings, mask)
            predicted_probabilities = self.model.convert_to_numpy(predicted_probabilities, mask)
        if self.model_type == "classification_binary":
            predictions = (
                predicted_probabilities >= self.optimal_cutoff if self.optimal_cutoff else None
            )
        else:
            # the head is sigmoid bounded, so the bounded output is the prediction
            # itself and there is no threshold to apply
            predictions = predicted_probabilities''',
    ),
    (
        "skip the cutoff search after training a regressor",
        '''        _, predicted_probabilities = self.forward(x_val)
        if self.residue_prediction_mode:''',
        '''        if self.model_type != "classification_binary":
            # an optimal cutoff is a classification idea, there is no threshold to find
            return
        _, predicted_probabilities = self.forward(x_val)
        if self.residue_prediction_mode:''',
    ),
]

if not os.path.exists(target):
    raise SystemExit(f"could not find {target} - pass the PALM checkout as the first argument")

source = open(target).read()
applied, already = [], []
for label, old, new in patches:
    if new in source:
        already.append(label)
    elif old in source:
        source = source.replace(old, new, 1)
        applied.append(label)
    else:
        raise SystemExit(f"could not apply {label!r}: PALM has changed, patch by hand")

if applied:
    backup = target + ".orig"
    if not os.path.exists(backup):
        open(backup, 'w').write(open(target).read())
        print(f"kept the original at {backup}")
    open(target, 'w').write(source)

for label in applied:
    print(f"  applied  {label}")
for label in already:
    print(f"  already  {label}")
print("\nnow run with dataset.task and predictor.model_type both set to regression")
