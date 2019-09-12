package tk.gustavo.pages;

import org.fluentlenium.core.annotation.PageUrl;
import org.fluentlenium.core.domain.FluentWebElement;
import static org.fluentlenium.core.filter.FilterConstructor.*;

@PageUrl("http://localhost:5000/")
public class HomePage extends BasePage {

    public FluentWebElement getArticleWithTitle(String title){
        FluentWebElement article = $(".article-title", withText(title)).first().dom().parent().dom().parent();
        return article;
    }

    public AboutPage clickAboutButton(){
        aboutButton.click();
        return aboutPage;
    }
}
