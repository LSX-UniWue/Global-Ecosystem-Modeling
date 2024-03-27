import numpy as np
import matplotlib.pyplot as plt


def create_biome_weighting_factor(biome_map, is_place_to_eval_mask):
    Class_names = {
        0: ['mid-latitude water-driven', '0x965635', 'MidL_W'],
        1: ['transitional energy-driven', '0xA5CC46', 'Trans_E'],
        2: ['boreal energy-driven', '0x44087C', 'Bor_E'],
        3: ['tropical', '0x4967D9', 'Tropic'],
        4: ['boreal temperature-driven', '0xcc76d1', 'Bor_T'],
        6: ['subtropical water-driven', '0xE8A76B', 'SubTr_E'],
        5: ['mid-latitude temperature-driven', '0x4A885B', 'MidL_T'],
        7: ['boreal water/temperature-driven', '0x5F3693', 'Bor_WT'],
        8: ['transitional water-driven', '0xA6EB99', 'Trans_W'],
        9: ['boreal water-driven', '0x8E71D5', 'Bor_W'],
        10: ['subtropical energy-driven', '0x296218', 'SubTr_E'],
        11: ['missing value', '0x808080', 'MISS'],
        12: ['missing value', '0x000000', 'MISS'],
    }
    # map to 0-11 range
    biome_map = biome_map.astype(int)

    biome_count = np.bincount(biome_map.flatten())

    # use the is_place_to_eval_mask to exclude places where we have no valid data, that is replace them with the missing-value class 11
    biome_map[is_place_to_eval_mask == False] = 11

    # get number of biomes
    biome_count = np.bincount(biome_map.flatten())

    # exlude missing-value biomes
    biome_count = biome_count[:-1]  # excludes class 11 and 12 (which is barren land and sea)
    num_biomes = len(biome_count)

    total_biome_count = np.sum(biome_count)

    # create biome weighting factor
    biome_weighting_factor = np.zeros(num_biomes)
    for i in range(num_biomes):
        biome_weighting_factor[i] = biome_count[i] / total_biome_count

    return biome_weighting_factor


# this is the biome mapping from Papagiannopoulou et al
biome_map = np.load("path_to/reconstructed_biomes.npy")
biome_map = biome_map[:720, :1440, 0]
# this is the mask where we have valid data according to the LSTM-paper's rules
is_place_to_eval_mask = np.load("path_to/eval_mask.npy")
is_place_to_eval_mask = is_place_to_eval_mask[:720, :1440]

plt.imshow(biome_map)
plt.colorbar()
plt.show()

biome_weighting_factor = create_biome_weighting_factor(biome_map, is_place_to_eval_mask)
