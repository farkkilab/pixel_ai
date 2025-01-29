import slideflow as sf
from slideflow.mil import mil_config

config = mil_config('attention_mil', lr=1e-3)



if __name__ == '__main__':
    P = sf.create_project(root='/scratch/project_2003009/he_space_slideflow',
                          slides='/scratch/project_2003009/space_he/adjacent_slide/',
                          annotations='/projappl/project_2003009/matias/pixel_ai/data/matched_data_space.csv')

    #P = sf.load_project('/scratch/project_2003009/public_he')

    #mpp_override = 1
    P.extract_tiles(tile_px=256, tile_um=128, num_threads=64)