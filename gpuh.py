

point_conf = 0.3
feedb_conf = 0.3
color_conf = 0.3

ja_point_conf = 0.3
ja_feedb_conf = 0.3
ja_color_conf = 0.3

emul_conf = 2

rri_conf = 0.2*4

op_conf = 0.5 * 3  # hidden, normal, expert

form_conf = 0.2*8
scaf_conf = 0.2*4

configurations = point_conf + feedb_conf + color_conf + ja_point_conf + ja_feedb_conf + ja_color_conf + emul_conf + rri_conf + op_conf + form_conf + scaf_conf
# configurations = 1

#
configurations = 0.3 + 0.3 + 0.3 + 8*0.3

# configurations = 3*0.2 + 0.5 + 0.04*2 + 0.5*2

# num_of_trains = 3 + 3 + 2 + 4 + 3 + 8 + 4
# print("num_of_trains:", num_of_trains)

configurations = 0.01 * 6

print(f"Number of trains: {configurations}")

frames = 100_000_000
# frames = 75_000_000
# frames = 50_000_000

seeds = 8
# seeds = 4
print(f"Number of seeds: {seeds}")

# ## one GPU
# fps = 300
fps = 580  # ssh jz
# fps = 500 # ssh pf

gpus_per_seed = 1
print(f"\n{gpus_per_seed} GPU")

seed_frames = frames
one_seed_time = 1_000_000 / (fps * 60 * 60)
print("train time (1M frames): {}h - {:d}d {:.0f}h".format(
    one_seed_time,
    int(one_seed_time // 24), one_seed_time % 24)
)

total_gpuh = configurations*seeds*gpus_per_seed*frames/(fps*60*60)
print("total gpu hours 1 gpups:", total_gpuh)

# ## half a GPU
#
# fps = 275
# fps = 370  # ssh jz
# # fps = 300 # ssh pf
# gpus_per_seed = 0.5
#
# print(f"\n{gpus_per_seed} GPU")
# one_seed = frames/(fps*60*60)
# print("train time: {}h - {:d}d {:.0f}h".format(one_seed, int(one_seed // 24), one_seed % 24))
#
# total_gpuh = configurations*seeds*gpus_per_seed*frames/(fps*60*60)
# print("total gpu hours 0.5 gpups:", total_gpuh)
#
# # ## 1/3 of a GPU
# fps = 250 # ssh jz 1/3
# # fps = 250 # ssh 1/3 pf
#
# gpus_per_seed = 0.33
# print(f"\n{gpus_per_seed} GPU")
#
# one_seed = frames/(fps*60*60)
# print("train time: {}h - {:d}d {:.0f}h".format(one_seed, int(one_seed // 24), one_seed % 24))
#
# total_gpuh = configurations*seeds*gpus_per_seed*frames/(fps*60*60)
# print("total gpu hours 0.33 gpups:", total_gpuh)
#
#
# # ## 1/4 of gpu
# # fps = 190 # ssh 1/4 pf
# #
# # gpus_per_seed = 0.25
# # print(f"\n{gpus_per_seed} GPU")
# #
# # one_seed = frames/(fps*60*60)
# # print("train time: {}h - {:d}d {:.0f}h".format(one_seed, int(one_seed // 24), one_seed % 24))
# #
# # total_gpuh = configurations*seeds*gpus_per_seed*frames/(fps*60*60)
# # print("total gpu hours 0.25 gpups:", total_gpuh)
