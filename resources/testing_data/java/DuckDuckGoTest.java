package tk.gustavo;

import org.fluentlenium.core.annotation.Page;
import org.junit.Test;
import tk.gustavo.pages.DuckDuckMainPage;

public class DuckDuckGoTest extends AbstractChromeTest{
    @Page
    private DuckDuckMainPage duckDuckMainPage;

    @Test
    public void titleOfDuckDuckGoShouldContainSearchQueryName() {
        String searchPhrase = "searchPhrase";

        goTo(duckDuckMainPage)
                .typeSearchPhraseIn(searchPhrase)
                .submitSearchForm()
                .assertIsPhrasePresentInTheResults(searchPhrase);
    }
}
