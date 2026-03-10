package com.guywithburrito.setfinder.card

fun Triple<SetCard, SetCard, SetCard>.isSet() : Boolean = SetCard.isSet(first, second, third)

data class SetCard(val shape: Shape, val pattern: Pattern, val count: Count, val color: Color) {
    enum class Shape {
        OVAL, DIAMOND, SQUIGGLE,
    }

    enum class Pattern {
        SOLID, SHADED, EMPTY,
    }

    enum class Count {
        ONE, TWO, THREE,
    }

    enum class Color {
        RED, GREEN, PURPLE,
    }

    companion object {
        fun isSet(card1: SetCard, card2: SetCard, card3: SetCard): Boolean {
            fun <T> numDistinct(block: SetCard.() -> T): Int =
                setOf(block(card1), block(card2), block(card3)).size

            return numDistinct { shape } != 2 && numDistinct { pattern } != 2 &&
                    numDistinct { count } != 2 && numDistinct { color } != 2
        }

        /**
         * Finds all valid sets in a list of cards.
         */
        fun findSets(cards: List<SetCard>): List<List<SetCard>> {
            val found = mutableListOf<List<SetCard>>()
            for (i in 0 until cards.size) {
                for (j in i + 1 until cards.size) {
                    for (k in j + 1 until cards.size) {
                        if (isSet(cards[i], cards[j], cards[k])) {
                            found.add(listOf(cards[i], cards[j], cards[k]))
                        }
                    }
                }
            }
            return found
        }
    }
}