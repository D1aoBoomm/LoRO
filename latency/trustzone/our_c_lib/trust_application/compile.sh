make clean
make

sudo rm /usr/lib/optee_armtz/8aaaf200-2450-11e4-abe2-0002a5d5c566.ta
sudo cp ./ta/8aaaf200-2450-11e4-abe2-0002a5d5c566.ta /usr/lib/optee_armtz/
sudo ./host/inference_demo
