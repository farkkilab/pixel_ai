import slideflow as sf



if __name__ == '__main__':
    #P = sf.create_project(root='/scratch/project_2003009/slideflow_space_he',
    #                      slides='/scratch/project_2003009/space_he/')
    P = sf.load_project('/scratch/project_2003009/slideflow_space_he')


    P.extract_tiles(tile_px=256, tile_um=128,num_threads=1)