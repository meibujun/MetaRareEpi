"""
metararepi.federated — Ray-based decentralised federated analysis.

Each biobank node runs as a Ray actor, computing local cumulants on its
private genotype data.  Only the (tiny) cumulant vectors are transmitted
to the aggregator — genotype data never leaves the node.
"""
