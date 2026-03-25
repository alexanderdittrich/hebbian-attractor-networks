from kandel.model.elman_recurrent import (
    ElmanRecurrentNeuralNetwork,
    HebbianElmanRecurrentNeuralNetwork,
)
from kandel.model.feed_forward import (
    FeedforwardNeuralNetwork,
    HebbianABCDNeuralNetwork,
)
from kandel.model.fully_connected_recurrent import (
    FullyConnectedRecurrentedNeuralNetwork,
    HebbianFullyConnectedRecurrentedNeuralNetwork,
)
from kandel.model.gated_recurrent_unit import (
    GRUNeuralNetwork,
    HebbianGRUNeuralNetwork,
)

networks = {
    FeedforwardNeuralNetwork.name: FeedforwardNeuralNetwork,
    HebbianABCDNeuralNetwork.name: HebbianABCDNeuralNetwork,
    ElmanRecurrentNeuralNetwork.name: ElmanRecurrentNeuralNetwork,
    HebbianElmanRecurrentNeuralNetwork.name: HebbianElmanRecurrentNeuralNetwork,
    FullyConnectedRecurrentedNeuralNetwork.name: FullyConnectedRecurrentedNeuralNetwork,
    HebbianFullyConnectedRecurrentedNeuralNetwork.name: HebbianFullyConnectedRecurrentedNeuralNetwork,
    GRUNeuralNetwork.name: GRUNeuralNetwork,
    HebbianGRUNeuralNetwork.name: HebbianGRUNeuralNetwork,
}
