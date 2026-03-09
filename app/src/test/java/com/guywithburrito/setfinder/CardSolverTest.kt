package com.guywithburrito.setfinder

import com.guywithburrito.setfinder.card.SetCard
import com.guywithburrito.setfinder.cv.CardFinder
import com.guywithburrito.setfinder.cv.CardUnwarper
import com.guywithburrito.setfinder.ml.TFLiteCardIdentifier
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.mockito.kotlin.mock

@RunWith(RobolectricTestRunner::class)
class CardSolverTest {

    @Test
    fun isSet_validSets_returnTrue() {
        val c1 = SetCard(SetCard.Shape.DIAMOND, SetCard.Pattern.SOLID, SetCard.Count.ONE, SetCard.Color.RED)
        val c2 = SetCard(SetCard.Shape.DIAMOND, SetCard.Pattern.SOLID, SetCard.Count.TWO, SetCard.Color.RED)
        val c3 = SetCard(SetCard.Shape.DIAMOND, SetCard.Pattern.SOLID, SetCard.Count.THREE, SetCard.Color.RED)
        
        assertThat(SetCard.isSet(c1, c2, c3)).isTrue()
    }

    @Test
    fun findSets_detectsCorrectSets() {
        val mockFinder: CardFinder = mock()
        val mockUnwarper: CardUnwarper = mock()
        val mockIdentifier: TFLiteCardIdentifier = mock()
        
        val detector = SetDetector(mockFinder, mockUnwarper, mockIdentifier)
        
        val c1 = SetCard(SetCard.Shape.DIAMOND, SetCard.Pattern.SOLID, SetCard.Count.ONE, SetCard.Color.RED)
        val c2 = SetCard(SetCard.Shape.DIAMOND, SetCard.Pattern.SOLID, SetCard.Count.TWO, SetCard.Color.RED)
        val c3 = SetCard(SetCard.Shape.DIAMOND, SetCard.Pattern.SOLID, SetCard.Count.THREE, SetCard.Color.RED)
        val c4 = SetCard(SetCard.Shape.OVAL, SetCard.Pattern.EMPTY, SetCard.Count.ONE, SetCard.Color.PURPLE)
        
        val cards = listOf(c1, c2, c3, c4)
        val sets = detector.findSets(cards)
        
        assertThat(sets).hasSize(1)
        assertThat(sets[0]).containsExactly(c1, c2, c3)
    }
}
