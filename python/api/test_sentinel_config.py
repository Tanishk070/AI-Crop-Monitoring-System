from sentinelhub import SHConfig

config = SHConfig()

# 👇 PASTE YOUR NEW CREDENTIALS HERE (DON'T SHARE THEM ANYWHERE)
config.sh_client_id = "sh-1f23a265-47d4-4ad9-b3ad-cccf4972c36e"
config.sh_client_secret = "osnR7VJ1QsAGH2pZtIT6COsKqocHuSmG"

config.save()
print("✅ Sentinel Hub credentials saved to local config.")
print("Client ID:", config.sh_client_id[:10] + "...")  # only partial, for sanity check
