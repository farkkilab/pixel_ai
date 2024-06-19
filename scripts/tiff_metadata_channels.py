import ipdb
import ome_types
xml = ome_types.from_tiff("/data/projects/sciset/registration/Sample_01.ome.tif")
names = ["DAPI","GFAP","Iba1-555","TUJ1-647",
"DAPI","p24-ahu-488","Nada","CD68-647",
"DAPI","TMEM119-Rbbad","CD11b-Cy3","PGT121Cy5",
"DAPI","HLA-DR","KC57PE","S100b",
]
for i in range(len(xml.images[0].pixels.channels)):
    if i >= len(names):
        break
    xml.images[0].pixels.channels[i].name = names[i]
f = open("./names.xml","w")
f.write(xml.to_xml())
f.close()
