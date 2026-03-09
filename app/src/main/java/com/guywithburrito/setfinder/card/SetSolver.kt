package com.guywithburrito.setfinder.card

/**
 * Interface for set-solving logic to allow for mocking in tests.
 */
interface SetSolver {
    /**
     * Finds all valid sets in a list of cards.
     */
    fun solve(cards: List<SetCard>): List<List<SetCard>>
}

/**
 * Standard implementation using the rules defined in SetCard.
 */
class DefaultSetSolver : SetSolver {
    override fun solve(cards: List<SetCard>): List<List<SetCard>> {
        val found = mutableListOf<List<SetCard>>()
        for (i in 0 until cards.size) {
            for (j in i + 1 until cards.size) {
                for (k in j + 1 until cards.size) {
                    if (SetCard.isSet(cards[i], cards[j], cards[k])) {
                        found.add(listOf(cards[i], cards[j], cards[k]))
                    }
                }
            }
        }
        return found
    }
}
