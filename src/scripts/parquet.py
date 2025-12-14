from datasets import load_dataset

ds = load_dataset("dylanmcguir3/insert-tool", split="train")
ds_filtered = ds.filter(lambda ex: ex["episode_index"] != 16)

# Save new dataset infos automatically
ds_filtered.push_to_hub(
    "dylanmcguir3/insert-tool",
    commit_message="Removed episode_index==16 and refreshed metadata",
)
