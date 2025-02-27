import argparse
import omegaconf
import pathlib

parser = argparse.ArgumentParser(prog='Metra')
parser.add_argument('--config')

args = parser.parse_args()
print(args)
config_folder = str(pathlib.Path(__file__).parent.resolve()) + '/configs'
config = config_folder + '/' + args.config
config = omegaconf.OmegaConf.load(config)
print(config)