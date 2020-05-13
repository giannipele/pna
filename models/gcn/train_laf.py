from __future__ import division
from __future__ import print_function

from models.gcn.layer import GCNLafLayer
from util.train import execute_train, build_arg_parser

# Training settings
parser = build_arg_parser()
parser.add_argument('--aggregation', type=str, default='mean', help='Type of LAF aggregation')
args = parser.parse_args()

execute_train(gnn_args=dict(nfeat=None,
                            nhid=args.hidden,
                            nodes_out=None,
                            graph_out=None,
                            dropout=args.dropout,
                            device=None,
                            first_conv_descr=dict(layer_type=GCNLafLayer, args=dict(function=args.aggregation)),
                            middle_conv_descr=dict(layer_type=GCNLafLayer, args=dict(function=args.aggregation)),
                            fc_layers=args.fc_layers,
                            conv_layers=args.conv_layers,
                            skip=args.skip,
                            gru=args.gru,
                            fixed=args.fixed,
                            variable=args.variable), args=args)
