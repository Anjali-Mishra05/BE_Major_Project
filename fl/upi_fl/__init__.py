"""Federated GBDT for UPI fraud detection - Phase II, Objectives 4 and 5.

This package is a Flower app. It trains a gradient-boosted decision tree model across
several simulated bank/POS clients, none of which shares raw transactions with the server.

Nothing here trains on, edits or replaces the Phase I artefacts. ``src/`` and ``models/``
are read-only inputs, and the centralized baseline in ``reports/baseline_metrics.json`` is
the comparison target: the point of Objective 5 is to show how close the federated model
gets to it without any client pooling its data.

Layout
------
``bins``        fixed, leak-free feature binning shared by server and clients
``gbdt``        the federated GBDT itself: histograms, split choice, leaf values, scoring
``schema.json`` the feature list and bin layout, bundled into the app so both sides agree
``client_app``  Flower ClientApp - holds a shard, sends only summed statistics
``server_app``  Flower ServerApp - drives the rounds, aggregates, owns the global model
"""
