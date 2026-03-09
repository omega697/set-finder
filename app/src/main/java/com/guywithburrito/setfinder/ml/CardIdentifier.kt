package com.guywithburrito.setfinder.ml

import android.content.Context
import android.graphics.Bitmap
import com.guywithburrito.setfinder.card.SetCard

/**
 * High-level interface for identifying a card from a pre-extracted chip.
 * Orchestrates Stage 2 (Filtering) and Stage 3 (Identification).
 */
interface CardIdentifier {
    /**
     * Returns an identified SetCard, or null if it's not a card or confidence is too low.
     */
    fun identifyCard(chip: Bitmap): SetCard?
    
    fun close()

    companion object {
        /**
         * Factory method to get the default modular TFLite implementation.
         */
        fun getInstance(context: Context): CardIdentifier {
            val filter = CardFilter.getInstance(context)
            val expert = CardExpert.getInstance(context)
            return TFLiteCardIdentifier(filter, expert)
        }
    }
}

/**
 * Orchestrator that delegates to CardFilter and CardExpert.
 */
class TFLiteCardIdentifier(
    private val filter: CardFilter,
    private val expert: CardExpert
) : CardIdentifier {

    override fun identifyCard(chip: Bitmap): SetCard? {
        // Stage 2: Filter
        if (!filter.isCard(chip)) {
            return null
        }
        
        // Stage 3: Expert Identification
        return expert.identify(chip)
    }

    override fun close() {
        filter.close()
        expert.close()
    }
}
