import slideflow as sf
from slideflow.mil import mil_config

config = mil_config('attention_mil', lr=1e-3)



if __name__ == '__main__':
    P = sf.create_project(root='/scratch/project_2003009/he_space_slideflow_cropped',
                          slides='/scratch/project_2003009/space_he/adjacent_slide_cropped/',
                          annotations='/projappl/project_2003009/matias/pixel_ai/data/matched_data_space.csv')

    #P = sf.load_project('/scratch/project_2003009/public_he')

    mpp_override = 10
    P.extract_tiles(tile_px=128, tile_um=1024, num_threads=64, mpp_override=mpp_override)