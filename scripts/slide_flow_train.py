import slideflow as sf
from slideflow.mil import mil_config

config = mil_config('attention_mil', lr=1e-3)



if __name__ == '__main__':
    P = sf.load_project('/scratch/project_2003009/public_he')

    full_dataset = dataset = P.dataset(tile_px=256, tile_um=128)
    print(dataset.summary())

    retccl = sf.build_feature_extractor('retccl',resize=256)

    # Generate features for a dataset
    dataset.generate_feature_bags(retccl, outdir='/scratch/project_2003009/public_he/retccl_features')

    # Split the dataset using three-fold, site-preserved cross-validation
    splits = full_dataset.kfold_split(
        k=3,
        labels='er_status_by_ihc'
    )

    # Train on each cross-fold
    for train, val in splits:
        P.train_mil(
            config=config,
            outcomes='er_status_by_ihc',
            train_dataset=train,
            attention_heatmaps=True,
            val_dataset=val,
            bags='/scratch/project_2003009/public_he/retccl_features'
        )
