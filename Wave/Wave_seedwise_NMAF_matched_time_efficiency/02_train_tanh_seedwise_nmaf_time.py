from seedwise_wave_common import PACKAGE_VERSION, train_all_seeds_seedwise_matched
if __name__ == "__main__":
    print(f"Package version: {PACKAGE_VERSION}")
    train_all_seeds_seedwise_matched("tanh")
