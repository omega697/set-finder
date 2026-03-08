package com.guywithburrito.setfinder

import com.guywithburrito.setfinder.tracking.TrackedCard
import com.guywithburrito.setfinder.card.SetCard
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.opencv.core.Point

class SetAnalyzerUnitTest {

    @Test
    fun findSets_findsCorrectSets() {
        // Create 3 cards that form a SET
        val c1 = SetCard(SetCard.Shape.OVAL, SetCard.Pattern.SOLID, SetCard.Count.ONE, SetCard.Color.RED)
        val c2 = SetCard(SetCard.Shape.OVAL, SetCard.Pattern.SOLID, SetCard.Count.TWO, SetCard.Color.RED)
        val c3 = SetCard(SetCard.Shape.OVAL, SetCard.Pattern.SOLID, SetCard.Count.THREE, SetCard.Color.RED)
        
        // One extra card that doesn't fit
        val c4 = SetCard(SetCard.Shape.DIAMOND, SetCard.Pattern.EMPTY, SetCard.Count.ONE, SetCard.Color.PURPLE)

        val tracked = listOf(
            TrackedCard(card = c1, bounds = emptyList()),
            TrackedCard(card = c2, bounds = emptyList()),
            TrackedCard(card = c3, bounds = emptyList()),
            TrackedCard(card = c4, bounds = emptyList())
        )

        val sets = mutableListOf<List<TrackedCard>>()
        for (i in 0 until tracked.size) {
            for (j in i + 1 until tracked.size) {
                for (k in j + 1 until tracked.size) {
                    val card1 = tracked[i].card
                    val card2 = tracked[j].card
                    val card3 = tracked[k].card
                    if (card1 != null && card2 != null && card3 != null &&
                        SetCard.isSet(card1, card2, card3)) {
                        sets.add(listOf(tracked[i], tracked[j], tracked[k]))
                    }
                }
            }
        }

        assertThat(sets).hasSize(1)
        assertThat(sets[0]).containsExactly(tracked[0], tracked[1], tracked[2])
    }
}
