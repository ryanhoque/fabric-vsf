# Instructions on autoencoder cost

1. Generate data. Get some `demos*.pkl` files from data collection (~500-1000 episodes) and run `python vismpc/scripts/predict.py --batch --input_img [file].pkl` to generate imagined data `.pkl` files.
2. Run `python format_vae_data.py [file1.pkl] [file2.pkl] ...` to generate a single `vae_data.pkl`.
3. Run `mkdir models imgs`
4. Run `python vae.py`. Ensure lines 38-39 make sense for the amount of data you're using (each number in the range is 128 data points).
5. See `imgs/` to visualize performance. Your trained model in `models/` will be ready for use by `vismpc/cost_functions.py`.
