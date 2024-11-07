import slideflow as sf
from slideflow.mil import mil_config

config = mil_config('attention_mil', lr=1e-3)



if __name__ == '__main__':
    P = sf.create_project(root='/scratch/project_2003009/public_he',
                          slides='/scratch/project_2003009/PKG-PTR-HGSOC/data/',
                          annotations='/scratch/project_2003009/PKG-PTR-HGSOC/PTRC-HGSOC_clinical_data.csv')

    #P = sf.load_project('/scratch/project_2003009/public_he')


    P.extract_tiles(tile_px=256, tile_um=128, num_threads=64)