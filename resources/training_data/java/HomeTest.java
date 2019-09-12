package tk.gustavo;

import org.fluentlenium.core.annotation.Page;
import org.fluentlenium.core.domain.FluentWebElement;
import org.junit.Before;
import org.junit.Test;
import tk.gustavo.pages.HomePage;

import static org.junit.Assert.assertThat;
import static org.junit.Assert.assertTrue;

public class HomeTest extends AbstractChromeTest {

    @Page
    private HomePage homePage;

    @Before
    public void setUp(){goTo(homePage).isAt();}

    @Test
    public void testArticle(){
        FluentWebElement article = homePage.getArticleWithTitle("Godammit");
        System.out.println(article.text());
        assertTrue(article.present());
    }

    @Test
    public void enterAboutPageThroughHomePage(){
        String url = homePage.clickAboutButton().url();
        assertTrue(url.contains("/about"));
    }
}
