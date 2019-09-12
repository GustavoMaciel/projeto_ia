package tk.gustavo;

import org.fluentlenium.core.annotation.Page;
import org.junit.Before;
import org.junit.Test;
import tk.gustavo.pages.LoginPage;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class LoginTest extends AbstractChromeTest{

    private final String email = "gm.nunes92@gmail.com";
    private final String password = "1234";

    @Page
    private LoginPage loginPage;

    @Before
    public void setUp(){
        getFluentControl().window().maximize();
        goTo(loginPage).isAt();
    }

    @Test
    public void tc001_loginWithValidCredentials(){
        boolean loggedIn = loginPage
                .login("gm.nunes92@gmail.com", "1234")
                .isLoggedIn();

        assertTrue(loggedIn);
    }

    @Test
    public void tc002_loginWithInvalidCredentials(){
        loginPage.login("test@test.com", "yadayada");
        assertTrue(loginPage.hasFlashAlert("//div[@class='alert alert-danger']"));
    }

    @Test
    public void tc003_loginWithInvalidFormatEmail(){
        boolean invalidFeedback = loginPage
                .login("test", password)
                .hasInvalidFeedback();

        assertTrue(invalidFeedback);
    }


    @Test
    public void tc004_loginWithoutEmail(){
        executeScript(
                "document.getElementsByName('email')[0].removeAttribute('required')"
        );

        boolean invalidFeedback = loginPage
                .fillPassword(password)
                .clickSubmitButton()
                .hasInvalidFeedback();
        assertTrue(invalidFeedback);
    }

    @Test
    public void tc005_loginWithoutPassword(){
        executeScript(
                "document.getElementsByName('password')[0].removeAttribute('required')"
        );

        boolean invalidFeedback = loginPage
                .fillEmail(email)
                .clickSubmitButton()
                .hasInvalidFeedback();
        assertTrue(invalidFeedback);
    }

    @Test
    public void tc006_accessRestrictedPageWithoutBeingLoggedIn(){
        goTo("http://localhost:5000/post/new");
        assertTrue(loginPage.hasFlashAlert("//div[@class='alert alert-info']"));
    }

    // Needs improvement
    @Test
    public void tc007_logoutFromService(){
        loginPage.login(email, password)
                .logout();
        assertFalse(loginPage.isLoggedIn());
    }

}
