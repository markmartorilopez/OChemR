# Copyright (c) 2022 rxn4chemistry - Mark Martori Lopez

{
	"dataset_params": { 
				"train_path" : "train/", 			# Path to store training samples.
				"labelled_path" : "labelled/", 		# Path where to store the labelled training images.
				"img_width" : 1024, 				# Width final image.
			    "img_height" : 1024,				# Height final image.
			    "molecules_sizes" : [10,12,14],		# Molecules bond size.
			    "molecules_rotations" : [0,30,330], # Rotation angle of molecules.
			    "num_molecules_per_reaction" : 12,	# Number of molecules per image.
			    "num_reactions_per_epoch" : 5,		# Number of images per epoch, each epoch has its own rotation and bond size params.
			    "epochs" : 3}						# Num of epochs.
}