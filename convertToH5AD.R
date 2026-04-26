path = ... # Dossier dans lequel la donnée est localisée. Exemple: path = r"(C:\Users\mayeu\Desktop\TRAVAIL\MVA\Cours\Projet médecin)"
setwd(path)

library("SeuratObject")
library("Seurat")

file_name = "dataset_single_cell_adrenal_for_MVA.RDS" # adapter file_name au nom du fichier .RDS
raw_data = readRDS(file_name)
print(raw_data)

library("SeuratDisk")
SaveH5Seurat(seurat_obj, filename = "single_cells.h5Seurat", overwrite = TRUE)

Convert("single_cells.h5Seurat", dest = "h5ad")